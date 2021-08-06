#include <math.h>
#include <random>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <cuda_runtime.h>

#define two_ppi (6.28318530717958648)
#define ppi (3.14159265358979324)
struct particle{
    float x, y, angle, tauR, tau;
    float Fx,Fy;
};
struct object{
    float x;
    float y;
    float angle;
    float Fx;
    float Fy;
    float torque;
};

__device__ float exponential_runtime(curandState *state, float tauR) {
  float U = curand_uniform(state);
  return -log(U)*tauR;
}
__device__ float uniform_runtime(curandState *state, float a, float b)
{
    float U = curand_uniform(state);
    return b+U*(a-b);
}
 //extern C
__global__ void initrand(curandState *state, const int N_ptcl) {
  int seed = 42;
  int offset = 0;
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N_ptcl;
       tid += blockDim.x * gridDim.x) {
    curand_init(seed, tid, offset, &state[tid]);
  }
}
__global__ void draw_runtimes(float *tauR,
    curandState *state,
    const int N_ptcl, const int N_active, const float alpha)
{ // for loop allows more particles than threads.
// N = N_ptcl + N_C+N_A
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N_ptcl;
tid += blockDim.x * gridDim.x) {
        if(tid<N_active)
        {tauR[tid] = exponential_runtime(&state[tid], alpha);}
        else{tauR[tid]=0;}
    }
}
void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

// initializing RNG for all threads with the same seed
// each state setup will be the state after 2^{67}*tid calls 
__global__ void initialize_prng(const int ptlsNum, 
        unsigned int seed, curandState *state)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum)
        curand_init(seed, tid, 0, &state[tid]) ;
}
__global__ void init_object(
    curandState *__restrict__ state,
    float *__restrict__ paAngle,
    float *__restrict__ pax,
    float *__restrict__ pay,
    const int N_passive,
    const int lsize,
    const float dist
)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<(int)(N_passive*N_passive/2))
    {
        curandState localState = state[tid] ;
        float angle = two_ppi*curand_uniform(&localState);
        pax[tid] = (float)(tid%(int)(N_passive/2))*dist*2-(float)lsize/2.+dist/2.-cosf(angle)+((int)tid/(int)(N_passive/2)&1)*dist;
        pay[tid] = (float)((int)tid/(int)(N_passive/2))*dist-(float)lsize/2.+dist/2.-sinf(angle);
        paAngle[tid] = angle;
        printf("tid : %d\tx : %f\ty : %f\t angle : %f\n",tid,pax[tid],pay[tid],paAngle[tid]);
        state[tid] = localState;
    }
    else if (tid<N_passive*N_passive)
    {
        curandState localState = state[tid] ;
        float angle = two_ppi*curand_uniform(&state[tid]);
        pax[tid] = (float)(tid%(int)(N_passive/2))*dist*2-(float)lsize/2.+dist/2.+(((int)tid/(int)(N_passive/2)+1)&1)*dist;
        pay[tid] = (float)((int)(tid-N_passive*N_passive/2)/(int)(N_passive/2))*dist-(float)lsize/2.+dist/2.;
        paAngle[tid] = angle;
        printf("tid : %d\tx : %f\ty : %f\t angle : %f\n",tid,pax[tid],pay[tid],paAngle[tid]);
        state[tid] = localState;
    }
}
__global__ void init_random_config(
    struct particle *__restrict__ ptls, 
    curandState *__restrict__ state,
    float *__restrict__ paAngle,
    float *__restrict__ pax,
    float *__restrict__ pay,   
    const int lsize, 
    const int N_ptcl,
    const int N_passive,
    const int N_active,
    const int N_body,
    const float alpha,
    const float dtheta
    )
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<N_active){
        curandState localState = state[tid] ;
        ptls[tid].x = uniform_runtime(&localState, -lsize/2., lsize/2.) ;
        ptls[tid].y = uniform_runtime(&localState, -lsize/2., lsize/2.) ;
        ptls[tid].angle = two_ppi*curand_uniform(&localState) ;
        ptls[tid].tauR = exponential_runtime(&localState,alpha);
        ptls[tid].tau = 0.0;
        state[tid] = localState ;
    }
    else if(tid<N_active+(int)(N_passive*N_passive*N_body/2)){
        int objnum = (int)((tid-N_active)/N_body);
        float angle = paAngle[objnum];
        int bodynum = tid-N_active-objnum*N_body-(int)((N_body-1)/2);
        ptls[tid].x = pax[objnum]+cosf(angle+bodynum*dtheta);
        ptls[tid].y = pay[objnum]+sinf(angle+bodynum*dtheta);
    }
    else{
        int objnum = (int)((tid-N_active)/N_body);
        float angle = paAngle[objnum];
        int bodynum = tid-N_active-objnum*N_body-(int)((N_body-1)/2);
        ptls[tid].x = pax[objnum]+cosf(angle+bodynum*dtheta);
        ptls[tid].y = pay[objnum]+sinf(angle+bodynum*dtheta);
    }
}

__global__ void particles_move(
    struct particle *ptls,
    curandState *state,
    float *patorque,
    float *pax,
    float *pay,
    const int lsize,       
    const float U0,                 // RTP swim speed
    const float dt,   // time step
    const float alpha,
    const int N_ptcl,
    const int N_passive,
    const int N_active,
    const int N_body,
    const float mu_active,
    const float mu_R_A,
    const float mu_R_C
    )
{

    // particle index
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    float tempx,tempy;
    if(tid < N_active)
    {
        ptls[tid].tau += dt;
        tempx = ptls[tid].x, tempy = ptls[tid].y;
        float tempAngle = ptls[tid].angle;
        if (ptls[tid].tau >= ptls[tid].tauR) {
            curandState localstate = state[tid];
            float deltat = ptls[tid].tau-ptls[tid].tauR;
            tempx += U0*deltat *cosf(tempAngle);
            tempy += U0*deltat *sinf(tempAngle);
            // the orientation needs to change in a discrete fashion due to
            // tumbling. pick a new orientation uniformly between 0 and 2pi
            tempAngle = curand_uniform(&localstate) * two_ppi;
            tempx += U0*(dt-deltat)*cosf(tempAngle);
            tempy += U0*(dt-deltat)*sinf(tempAngle);
            // reset time since last tumble to zero.
            ptls[tid].tau = (dt-deltat);
            // after tumbling, need to draw a new tumbling time.
            ptls[tid].tauR = exponential_runtime(&localstate, alpha);
            ptls[tid].angle = tempAngle;
            //printf("tid : %d\tx : %f\ty : %f\tangle : %f\n",tid,ptls[tid].tau,ptls[tid].tauR,ptls[tid].angle);
            state[tid] =localstate;
        }
        else{
            tempx += U0*dt*cosf(tempAngle);
            tempy += U0*dt*sinf(tempAngle);
        }
        tempx += mu_active*(ptls[tid].Fx)*dt;
        tempy += mu_active*(ptls[tid].Fy)*dt;
        if(tempx>(float)lsize/2.) tempx -= (float)lsize;
        if(tempx<-(float)lsize/2.) tempx += (float)lsize;
        if(tempy>(float)lsize/2.) tempy -= (float)lsize;
        if(tempy<-(float)lsize/2.) tempy += (float)lsize;
        ptls[tid].x = tempx;
        ptls[tid].y = tempy;
        
        //if(tid == 3)printf("4th particle x : %f\t y:%f \tFx:%f\tFy:%f\n",tempx,tempy,
        //ptls[tid].Fx,ptls[tid].Fy);
    }
    else if(tid<N_active+(int)(N_passive*N_passive*N_body/2))
    {
        int objnum = (int)((tid-N_active)/N_body);
        int centernum = N_active+objnum*N_body+(int)((N_body-1)/2);
        float dtheta = patorque[objnum]*dt;
        tempx = ptls[tid].x, tempy = ptls[tid].y;
        float dy = tempy-ptls[centernum].y;
        float dx = tempx-ptls[centernum].x;
        tempx  += -dy*dtheta;
        tempy  += dx*dtheta;
        if(tempx>(float)lsize/2.) tempx -= (float)lsize;
        if(tempx<-(float)lsize/2.) tempx += (float)lsize;
        if(tempy>(float)lsize/2.) tempy -= (float)lsize;
        if(tempy<-(float)lsize/2.) tempy += (float)lsize;
        ptls[tid].x = tempx;
        ptls[tid].y = tempy;
        //printf("tid : %d\tx : %f\ty : %f\t\n",tid-N_active,ptls[tid].x,ptls[tid].y);
    }
    else if(tid<N_ptcl)
    {
        int objnum = (int)((tid-N_active)/N_body);
        float dtheta = patorque[objnum]*dt;
        tempx = ptls[tid].x, tempy = ptls[tid].y;
        float dy = tempy-pay[objnum];
        float dx = tempx-pax[objnum];
        tempx  += -dy*dtheta;
        tempy  += dx*dtheta;
        if(tempx>(float)lsize/2.) tempx -= (float)lsize;
        if(tempx<-(float)lsize/2.) tempx += (float)lsize;
        if(tempy>(float)lsize/2.) tempy -= (float)lsize;
        if(tempy<-(float)lsize/2.) tempy += (float)lsize;
        ptls[tid].x = tempx;
        ptls[tid].y = tempy;
        //printf("tid : %d\tx : %f\ty : %f\t\n",tid-N_active,ptls[tid].x,ptls[tid].y);
    }
// need to tumble 
}
__global__ void force(
    struct particle *ptls,
    int *cellHead,
    int *cellTail,
    float *torque,
    float *pax,
    float *pay,
    const int lsize,
    const float lamb,
    const int N_ptcl,
    const int N_passive,
    const int N_active,
    const int N_body
)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    float Fx =0.0 ,Fy=0.0;
    float dx,dy,dl;
    if(tid < N_active) 
    {
        for(int i = N_active; i<N_ptcl; i++)
        {
            dx = ptls[tid].x-ptls[i].x;
            dy = ptls[tid].y-ptls[i].y;
            if      (dx > (float)lsize/2)  {dx -= lsize;}
            else if (dx < -(float)lsize/2) {dx += lsize;}
            if      (dy > (float)lsize/2)  {dy -= lsize;}
            else if (dy < -(float)lsize/2) {dy += lsize;}
            dl = dx*dx+dy+dy;
            if(dl<=1.0)
            {
                Fx += lamb*dx;
                Fy += lamb*dy;
            }
        }
        ptls[tid].Fx=Fx;
        ptls[tid].Fy=Fy;
    }
    else if(tid<N_active+(int)(N_passive*N_passive*N_body/2))
    {
        int objnum = (int)((tid-N_active)/N_body);
        int centernum = N_active+objnum*N_body+(int)((N_body-1)/2);
        float x = ptls[tid].x, y = ptls[tid].y;
        for(int a=(int)x-1; a<=(int)x+1; a++) {
            for(int b=(int)y-1; b<=(int)y+1; b++) {
                // zz : index for neighboring cells
                int zz = (a+(int)(lsize/2.0))%lsize + ((b+(int)(lsize/2.0))%lsize)*lsize ;
                for(int k=cellHead[zz]; k<=cellTail[zz]; k++) {
                    // loop over particles in the cell zz
                    dx = (x-ptls[k].x) ;
                    if(dx>lsize/2.) dx = dx-lsize ;
                    else if(dx<-lsize/2.) dx = dx+lsize;
                    dy = (y-ptls[k].y) ;
                    if(dy>lsize/2.) dy = dy-lsize ;
                    else if(dy<-lsize/2.) dy = dy+lsize;
                    if(dx*dx+dy*dy <= 1.0) {
                        Fx += lamb*dx;
                        Fy += lamb*dy;
                    }
                }
            }
        }
        
        dx = x-ptls[centernum].x;
        dy = y-ptls[centernum].y;
        //obj1.Fx += ptls[tid].Fx_A;
        //obj1.Fy += ptls[tid].Fy_A;
        //ptls[tid].Fy_C = dx*ptls[tid].Fy_A-dy*ptls[tid].Fx_A;
        torque[tid-N_active]=dx*Fy-dy*Fx;
        //printf("tid : %d \tdx:%f\tdy:%f\tFy:%f\tFx:%f\n",tid,dx,dy,Fy,Fx);
    }
    else if(tid<N_ptcl)
    {
        int objnum = (int)((tid-N_active)/N_body);
        float x = ptls[tid].x, y = ptls[tid].y;
        for(int a=(int)x-1; a<=(int)x+1; a++) {
            for(int b=(int)y-1; b<=(int)y+1; b++) {
                // zz : index for neighboring cells
                int zz = (a+(int)(lsize/2.0))%lsize + ((b+(int)(lsize/2.0))%lsize)*lsize ;
                for(int k=cellHead[zz]; k<=cellTail[zz]; k++) {
                    // loop over particles in the cell zz
                    dx = (x-ptls[k].x) ;
                    if(dx>lsize/2.) dx = dx-lsize ;
                    else if(dx<-lsize/2.) dx = dx+lsize;
                    dy = (y-ptls[k].y) ;
                    if(dy>lsize/2.) dy = dy-lsize ;
                    else if(dy<-lsize/2.) dy = dy+lsize;
                    if(dx*dx+dy*dy <= 1.0) {
                        Fx += lamb*dx;
                        Fy += lamb*dy;
                    }
                }
            }
        }
        dx = x-pax[objnum];
        dy = y-pay[objnum];
        //obj2.Fx += ptls[tid].Fx_A;
        //obj2.Fy += ptls[tid].Fy_A;
        //ptls[tid].Fy_C = dx*ptls[tid].Fy_A-dy*ptls[tid].Fx_A;
        torque[tid-N_active]=dx*Fy-dy*Fx;
        //printf("tid : %d \tdx:%f\tdy:%f\tFy:%f\tFx:%f\n",tid,dx,dy,Fy,Fx);
    }
    
}

// make a table "cell[i]" for the cell index for a particle i
__global__ void find_address(struct particle *ptls, 
        const int lsize, const int ptlsNum, int *cell)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
         cell[tid] = ((int)(ptls[tid].x+lsize/2.0))%lsize 
                    + lsize*(((int)(ptls[tid].y+lsize/2.0))%lsize) ;
    }
}

// make tables "cellHead[c]" and "cellTail[c]" for the index 
// of the first and the last praticle in a cell c
// empty cells are not updated
__global__ void cell_head_tail(int ptlsNum, int *cell, 
        int *cellHead, int *cellTail)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        if(tid==0) cellHead[cell[tid]] = tid ;
        else {
            if(cell[tid]!=cell[tid-1]) cellHead[cell[tid]] = tid ;
        }
        if(tid==ptlsNum-1) cellTail[cell[tid]] = tid ;
        else {
            if(cell[tid]!=cell[tid+1]) cellTail[cell[tid]] = tid ;
        }
    }
}


void linked_list(struct particle *ptls, const int lsize, 
        const int N_ptcl,const int N_active, const int cllsNum, int *cell,  
        int *head, int *tail, int nBlocks, int nThreads)
{
    // cell[ptl] = cell index of a particle
    find_address<<<nBlocks, nThreads>>>(ptls, lsize, N_ptcl, cell);
    // sort particles w.r.t the cell index
    thrust::sort_by_key(thrust::device_ptr<int>(cell),
                thrust::device_ptr<int>(cell)+N_active,
                thrust::device_ptr<struct particle>(ptls));
    thrust::fill(thrust::device_ptr<int>(head),
            thrust::device_ptr<int>(head)+cllsNum, 0);
    thrust::fill(thrust::device_ptr<int>(tail),
            thrust::device_ptr<int>(tail)+cllsNum, -1);
    // find the first (head) and the last (tail)  particle indices in each cell
    // head = -1 and tail = 0 for the empty cell
    cell_head_tail<<<nBlocks, nThreads>>>(N_active, cell, head, tail);
}
__global__ void torque_object(float *torque, float *patorque,float *paAngle, 
const int N_passive, const int N_body, const float mu_R_A,const float mu_R_C)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<(int)(N_passive*N_passive/2))
    {
        float temptorque=0;
        float tempAngle=paAngle[tid];
        for(int i = 0; i<N_body;i++)
        {
            temptorque += torque[tid*N_body+i];
        }
        //printf("tid : %d \t torque : %f\n",tid,temptorque);
        temptorque *= mu_R_A;
        tempAngle += temptorque;
        if(tempAngle>two_ppi)tempAngle -=two_ppi;
        if(tempAngle<0)tempAngle += two_ppi;
        paAngle[tid]=tempAngle;
        patorque[tid] = temptorque;
    }
    else if (tid<N_passive*N_passive){
        float temptorque=0;
        float tempAngle=paAngle[tid];
        for(int i = 0; i<N_body;i++)
        {
            temptorque += torque[tid*N_body+i];
        }
        //printf("tid : %d \t torque : %f\n",tid,temptorque);
        //printf("torque: %f\n",temptorque);
        temptorque *= mu_R_C;
        tempAngle += temptorque;
        if(tempAngle>two_ppi)tempAngle -=two_ppi;
        if(tempAngle<0)tempAngle += two_ppi;
        paAngle[tid]=tempAngle;
        patorque[tid] = temptorque;
    }
}
/*void get_orderParameter(struct particle *ptls, 
        const int N_ptcl, float *vx, float *vy, 
        float *odx, float *ody, const int nBlocks, const int nThreads)
{
    orderParameter<<<nBlocks, nThreads>>>(ptls, N_ptcl, vx, vy);
    *odx = thrust::reduce(thrust::device_ptr<float>(vx),
            thrust::device_ptr<float>(vx)+ptlsNum, 0.0, 
            thrust::plus<float>());
    *ody = thrust::reduce(thrust::device_ptr<float>(vy),
            thrust::device_ptr<float>(vy)+ptlsNum, 0.0, 
            thrust::plus<float>());
}*/