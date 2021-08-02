#define interactionRange2 (1.0) 
#define MaxThreads (256)

#include "activefluid_multi.c"
#include "stdlib.h"
#include <time.h>
#include <random>
#include <fstream>
#include <iostream>
int main(int argc, char *argv[])
{
    if(argc!=10) error_output("command L tmax delt") ;
    time_t start, end;
    const double  rho   = 800.0;
    const int    Lsize  = atoi(argv[1]);
    const int    tmax   = atoi(argv[2]);
    const double dt     = atof(argv[3]);
    const int    record = atof(argv[4]);
    const int    N_body = atoi(argv[5]);
    const double lamb   = atof(argv[6]);
    const double dist   = atof(argv[7]);
    const double mu_R_A = atof(argv[8])/360.0*two_ppi/rho;
    const double mu_R_C = atof(argv[9])/360.0*two_ppi/rho;
    const double mu_active = 1.0;
    const double alpha = 1.0;
    //const int    N_C    = atoi(argv[5]);
    //const int    N_A    = atoi(argv[6]);
    const double  U0  = 1.0;

    // total number of particles
    const  int   N_active = (int)(Lsize*Lsize*rho);
    const  int   N_passive = (int)(Lsize/dist);
    const  int   N_ptcl = N_active+N_passive*N_body*N_passive ;
    const  double dtheta = ppi/N_body;
    // total number of cells
    const  int   cllsNum    = Lsize*Lsize ;

    // grid dimension
    const int nThreads = (MaxThreads<N_ptcl)? MaxThreads : N_ptcl;
    const int nBlocks  = (N_ptcl+nThreads-1)/nThreads;
    std::cout<<N_ptcl<<'\n';
    double *paAngle, *paTorque, *pax, *pay;
    double *AngleHost;
    cudaMalloc(&paAngle, sizeof(double)*N_passive*N_passive);
    cudaMalloc(&paTorque, sizeof(double)*N_passive*N_passive);
    cudaMalloc(&pax, sizeof(double)*N_passive*N_passive);
    cudaMalloc(&pay, sizeof(double)*N_passive*N_passive);
    AngleHost = (double *)malloc(sizeof(double)*N_passive*N_passive);
    // VicsekParticle in the device
    struct particle *devPtls;
    cudaMalloc(&devPtls, sizeof(struct particle)*N_ptcl) ;
    // auxiliary memory for linked lists
    // linked list is managed with the THRUST library
    // corresponding device memory
    int *devCell, *devHead, *devTail ;
    double *devtorque;
    cudaMalloc(&devCell, sizeof(int)*N_ptcl);
    cudaMalloc(&devHead, sizeof(int)*cllsNum);
    cudaMalloc(&devTail, sizeof(int)*cllsNum);
    cudaMalloc(&devtorque, sizeof(double)*(N_passive*N_passive*N_body));
    

/* // temporary angle variable
    float *devAngTmp;
    cudaMalloc(&devAngTmp, sizeof(double)*ptlsNum);

    // order parameter measurement
    float *devVx, *devVy;
    cudaMalloc(&devVx, sizeof(float)*ptlsNum);
    cudaMalloc(&devVy, sizeof(float)*ptlsNum);
*/
    // set the PRNG seed with the device random number
    curandState *devStates;
    cudaMalloc(&devStates, sizeof(curandState)*N_ptcl);
    initrand<<<nBlocks,nThreads>>>(devStates, N_ptcl) ;
        // random initial configuration
        init_object<<<nBlocks,nThreads>>>(devStates,paAngle, pax, pay, N_passive,Lsize,dist);
        init_random_config<<<nBlocks,nThreads>>>(devPtls, devStates, paAngle, pax, pay, Lsize, N_ptcl,N_passive, N_active,N_body,alpha,dtheta) ;
        //init_passive_particle<<<nBlocks,nThreads>>>(devPtls, paAngle, pax, pay, Lsize, N_passive, N_active,N_body,dist) ;
        start = time(NULL);
        std::ofstream out;
        out.open("test.csv");
        for(double t=0.0; t<=tmax; t +=dt) {
            // position and angle update
             // linked list
            linked_list(devPtls, Lsize, N_ptcl,N_active, cllsNum, devCell, devHead, devTail,nBlocks, nThreads);
            force<<<nBlocks,nThreads>>>(devPtls,devHead,devTail,devtorque,pax,pay,Lsize,lamb,N_ptcl,N_passive,N_active,N_body);
            torque_object<<<nBlocks,nThreads>>>(devtorque, paTorque, paAngle,N_passive,N_body,mu_R_A,mu_R_C);
            particles_move<<<nBlocks, nThreads>>>(devPtls,devStates,paAngle,pax,pay,Lsize,U0,dt,alpha,
                N_ptcl,N_passive,N_active,N_body,mu_active,mu_R_A,mu_R_C);
            if((int)t%record==0)
            {
                cudaMemcpy(AngleHost,paAngle,sizeof(double)*N_passive*N_passive,cudaMemcpyDeviceToHost);
                for(int i = 0; i<(int)N_passive*N_passive/2;i++)
                {
                    out<<AngleHost[i]<<',';
                    out<<AngleHost[i+(int)N_passive*N_passive/2]<<',';
                }
                out<<'\n';
            }
        }
        end= time(NULL);
        std::cout<<(end-start)<<'\n';
    //}

    cudaFree(devPtls) ; cudaFree(devStates);  
    cudaFree(devCell) ; cudaFree(devHead) ; cudaFree(devTail) ; 
    cudaFree(paAngle); cudaFree(pax);cudaFree(pay);cudaFree(paTorque);
    free(AngleHost);
}