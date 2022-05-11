#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

#define LENGTH 100  // HEIGHT of boundary box
#define WIDTH 200   // WIDTH of boundary box
#define DEPTH 400   // DEPTH of boundary box
#define NUMBODIES 1000 // Number of Balls
#define MASS 1  // Mass of each body
#define RADIUS 0.5  // Radius of Ball
#define DELTA 0.01  // Delta T
#define SIMULATION_STEPS 720000 // Number of simulation steps

int numthreads;

class Particle
{

public:

    int collision=0; // particle-particle collision flag;
    double cen_x,cen_y,cen_z; //position of centers
    double vel_x,vel_y,vel_z; //velocity components
    double for_x,for_y,for_z; //force components
    double hvel_x,hvel_y,hvel_z; //half-velocity components
    double tvel_x,tvel_y,tvel_z; //temporary-velocity components(used while chcking particle-particle collision)

    Particle():cen_x(0.0),cen_y(0.0),cen_z(0.0),vel_x(0.0),vel_y(0.0),vel_z(0.0),hvel_x(0.0),hvel_y(0.0),hvel_z(0.0){}
    void updatehalfvel();
    void updatepos();
    void updatevel();
    void resetforce();
    void checkcollision(Particle& p);
    void resettempvelocity();
    void uvelC();
    void load(ifstream& inf);
    void save(ofstream& of);
    void wallCollision();
};


//Distance between two particles
double distance(Particle& p1, Particle& p2)
{
  double x = p2.cen_x - p1.cen_x;
  double y = p2.cen_y - p1.cen_y;
  double z = p2.cen_z - p1.cen_z;
  return x*x + y*y + z*z;
}

//Resetting temp velocities after updating them to actual velocity components
void Particle::resettempvelocity()
{
  this->tvel_x = 0.0;
  this->tvel_y = 0.0;
  this->tvel_z = 0.0;
  this->collision = 0;
}

// Updating actual velocities to temp velocities(if updated)
void Particle::uvelC()
{
  if(this->collision)
  {
    this->vel_x = this->tvel_x;
    this->vel_y = this->tvel_y;
    this->vel_z = this->tvel_z;
  }
}


// Update Half velocity
void Particle::updatehalfvel()
{
  this->hvel_x = this->vel_x + (this->for_x * DELTA)/ (2 * MASS);
  this->hvel_y = this->vel_y + (this->for_y * DELTA)/ (2 * MASS);
  this->hvel_z = this->vel_z + (this->for_z * DELTA)/ (2 * MASS);

}

// Update Velocities
void Particle::updatevel()
{
  this->vel_x = this->hvel_x + (this->for_x * DELTA) / (2 * MASS);
  this->vel_y = this->hvel_y + (this->for_y * DELTA) / (2 * MASS);
  this->vel_z = this->hvel_z + (this->for_z * DELTA) / (2 * MASS);
}

// Resetting force
void Particle::resetforce()
{
  this->for_x = 0.0;
  this->for_y = 0.0;
  this->for_z = 0.0;
}


// Update position
void Particle::updatepos()
{
  this->cen_x = this->cen_x + this->hvel_x * DELTA;
  this->cen_y = this->cen_y + this->hvel_y * DELTA;
  this->cen_z = this->cen_z + this->hvel_z * DELTA;
}

// Check collision
void checkcollision(Particle* &p)
{
     #pragma omp parallel for collapse(2)
     for (int i=0;i<NUMBODIES;i++)
     {
         for(int j=0;j<NUMBODIES;j++)
         {
             if(j>i && p[i].collision==0 && p[j].collision==0) 
             {
                 double dist = distance(p[i], p[j]);
                 dist+=1e-7;
                 dist=sqrt(dist);
                 double x = p[j].cen_x - p[i].cen_x;
                 double y = p[j].cen_y - p[i].cen_y;
                 double z = p[j].cen_z - p[i].cen_z;
                 double relv= (((p[j].vel_x - p[i].vel_x) * x)+((p[j].vel_y - p[i].vel_y) * y)+((p[j].vel_z - p[i].vel_z) * z))/ dist;    
                 double vx = -p[j].vel_x + p[i].vel_x;
                 double vy = -p[j].vel_y + p[i].vel_y;
                 double vz = -p[j].vel_z + p[i].vel_z;
                 double vmag = vx*vx+vy*vy+vz*vz;
                 vmag=sqrt(vmag);
                 double f1=p[j].vel_x*x+p[j].vel_y*y+p[j].vel_z*z,f2=(p[i].vel_x)*x+(p[i].vel_y)*y+(p[i].vel_z)*z; 
                 if(dist<2*RADIUS && (vmag>0 && (x*vmag==vx*dist) && (y*vmag==vy*dist) && (z*vmag==vz*dist) && (dist/abs(relv) < DELTA)))
                 {
                    #pragma omp critical
                    {
	                     p[i].collision = 1;
	                     p[i].tvel_x = p[i].vel_x+(f1-f2)*(x/(dist*dist));
	                     p[i].tvel_y = p[i].vel_y+(f1-f2)*(y/(dist*dist));
	                     p[i].tvel_z = p[i].vel_z+(f1-f2)*(z/(dist*dist));
	                     p[j].collision = 1;
	                     p[j].tvel_x = p[j].vel_x+(f2-f1)*(x/(dist*dist));
	                     p[j].tvel_y = p[j].vel_y+(f2-f1)*(y/(dist*dist));
	                     p[j].tvel_z = p[j].vel_z+(f2-f1)*(z/(dist*dist));
                    }
                 }
             }
         }
     }
}

//Check for wall collisions
void Particle::wallCollision(){

  if((this->cen_x + RADIUS) > WIDTH)
  {
    double del = this->cen_x + RADIUS - WIDTH;
    this->cen_x = WIDTH - del - RADIUS;
    this->vel_x = -this->vel_x;
  }
  else if((this->cen_x - RADIUS) < 0)
  {
    double del = RADIUS - this->cen_x;
    this->cen_x = RADIUS + del;
    this->vel_x = -this->vel_x;
  }

  if((this->cen_y + RADIUS) > LENGTH)
  {
    double del = this->cen_y + RADIUS - LENGTH;
    this->cen_y = LENGTH- del - RADIUS;
    this->vel_y = -this->vel_y;
  }
  else if(this->cen_y - RADIUS < 0)
  {
    double del = RADIUS - this->cen_y;
    this->cen_y = RADIUS + del;
    this->vel_y = -this->vel_y;
  }

  if(this->cen_z + RADIUS > DEPTH)
  {
    double del = this->cen_z + RADIUS - DEPTH;
    this->cen_z = DEPTH- del - RADIUS;
    this->vel_z = -this->vel_z;
  }
  else if(this->cen_z - RADIUS < 0)
  {
    double del = RADIUS - this->cen_z;
    this->cen_z = RADIUS + del;
    this->vel_z = -this->vel_z;
  }
}


//add net force on the particle
void addforce(Particle* &p)
{
    double fx[NUMBODIES]={0.0}, fy[NUMBODIES]={0.0}, fz[NUMBODIES]={0.0};
    #pragma omp parallel for collapse(2) reduction(+: fx, fy, fz)
    for (int i=0;i<NUMBODIES;i++)
    {
        for(int j=0;j<NUMBODIES;j++)
        {
            if(j!=i) 
            {
                double dist = distance(p[i], p[j]);
                double force = 0.0;
                //if(moddist != 0.0)
                dist += 1e-7;
                force = (MASS * MASS)/ dist;
                double x = p[j].cen_x - p[i].cen_x;
                double y = p[j].cen_y - p[i].cen_y;
                double z = p[j].cen_z - p[i].cen_z;
                fx[i] += (force * x)/ sqrt(dist);
                fy[i] += (force * y)/ sqrt(dist);
                fz[i] += (force * z)/ sqrt(dist);
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < NUMBODIES; ++i)
    {
        p[i].for_x = fx[i];
        p[i].for_y = fy[i];
        p[i].for_z = fz[i];
    }
}

// function for each simulation step
void simulation(Particle* particles)
{
    omp_set_num_threads(numthreads);
    checkcollision(particles);
    addforce(particles);
    #pragma omp parallel for
    for (int i=0;i<NUMBODIES;i++)
    {
        particles[i].uvelC();
        particles[i].updatehalfvel();
        particles[i].updatepos();
        //particles[i].uvelC();
        particles[i].updatevel();
        particles[i].wallCollision();
        particles[i].resetforce();
        particles[i].resettempvelocity();
    }

}


void Particle::save(ofstream& of)
{
    of.write(reinterpret_cast<char *>(&(this->cen_x)), sizeof(this->cen_x));
    of.write(reinterpret_cast<char *>(&(this->cen_y)), sizeof(this->cen_y));
    of.write(reinterpret_cast<char *>(&(this->cen_z)), sizeof(this->cen_z));
}


void Particle::load(ifstream& inf)
{
  inf.read((char*)&cen_x, sizeof(cen_x));
  inf.read((char*)&cen_y, sizeof(cen_y));
  inf.read((char*)&cen_z, sizeof(cen_z));
}

void writeBinary(ofstream& of, Particle* particles)
{
  for(int i = 0 ; i < NUMBODIES ; i++)
  {
    particles[i].save(of);
  }
}

void readBinary(ifstream& inf, Particle* particles)
{
  for(int i = 0 ; i < NUMBODIES ; i++)
  {
    particles[i].load(inf);
  }
}


int main( int argc,char* argv[])
{
    numthreads = atoi(argv[2]);
    const char* out = argv[1];
    cout << "Number of threads: "<< numthreads << endl;
    Particle* particles= new Particle[NUMBODIES];
    int bo=0;
    char *ax,*by,*cz;
    if (FILE *fp = fopen("Trajectory.txt", "r"))
    {
        char buf[1024];
        for(int i=0;i<8;i++)
        {
            fgets(buf, sizeof(buf), fp);
        }
        while(fgets(buf, sizeof(buf), fp)!=NULL && bo<NUMBODIES) 
        {
            if(strcmp(buf,"\n"  ) != 0 && strcmp(buf,"\r\n") != 0 && strcmp(buf,"\0"  ) != 0)
            {
                ax = strtok(buf, "\t");
                by = strtok(NULL, "\t");
                cz = strtok(NULL, "\t");
                particles[bo].cen_x=atof(ax);
                particles[bo].cen_y=atof(by);
                particles[bo].cen_z=atof(cz);
                bo++;
            }    
        }
             
        fclose(fp);
    }
    remove(out);
    ofstream myfile;
    myfile.open(out, ios::binary | ios::out | ios::app);
    writeBinary(myfile, particles);
    double start = omp_get_wtime();

    for(int i = 1 ; i <= SIMULATION_STEPS ; i++)
    {
        double st = omp_get_wtime();
        simulation(particles);
        if(i%100 == 0)
        {
          cout << "Time for step " << i << " : " << omp_get_wtime()-st << endl;
          writeBinary(myfile, particles);
        }
    }
    cout << "\nSimulation Finished.." << endl;

    cout << "Time: " << omp_get_wtime()-start << endl;
}
























