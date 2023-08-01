#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <random>
#include <cmath>
#include <omp.h>
#define NUM_THREADS 8


struct Ptcl
{
	int pos,spin;
};

int main(int argc, char **argv)
{
	int i,j,k,n,p,L2,N,pos,spin,right,left,up,down;
	int temp,cnt1,cnt2;
	double prob,prob_r,prob_l,prob_move,prob_move2,prob_flip,exponent,t;
	double avg_angle,totalvx,totalvy,m,mag,mag2,mag4;
	char confile[100],obsfile[100],rhofile[100],magfile[100];	
	omp_set_num_threads(NUM_THREADS);

	// Parse arguments
	if (argc<7){
		std::cout << "Usage: ./aim2d Lx Ly beta rho e tf FIRST\n";
		std::exit(1);
	}
	
	int Lx = atoi(argv[1]);
	int Ly = atoi(argv[2]);
	double beta = atof(argv[3]);
	double rho = atof(argv[4]);
	double e = atof(argv[5]);
	double tf = atof(argv[6]);
	int FIRST = atoi(argv[7]);


	// Declare parameters
	const int cnter1 = 1000;
	const int cnter2 = 10000;
	const double dt = 1/(4.0+exp(beta));
	prob_move = 2.0*dt;
	prob_move2 = 2.0*prob_move;
	L2 = Lx*Ly;
	N = L2*rho;
	t = 0;
	cnt1 = cnt2 = 0;

	// Set random number generators
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> uni(0,1);
	//std::uniform_real_distribution<double> symuni(-3.141592,3.141592);

	// Set parallel random number generator
	std::vector< std::default_random_engine > generators;
	generators.reserve(NUM_THREADS);
	for (i=0;i<NUM_THREADS;++i) generators.emplace_back(std::default_random_engine(rd()));

	// Set filestream
	sprintf(confile,"2D/Lx%dLy%dbeta%.3frho%.3fconf",Lx,Ly,beta,rho);
	sprintf(obsfile,"2D/Lx%dLy%dbeta%.3frho%.3fe%.2fobs",Lx,Ly,beta,rho,e);
	sprintf(rhofile,"2D/Lx%dLy%dbeta%.3frho%.3fe%.2fdensity",Lx,Ly,beta,rho,e);
	sprintf(magfile,"2D/Lx%dLy%dbeta%.3frho%.3fe%.2fmag",Lx,Ly,beta,rho,e);
	std::ofstream obsoutput;
	std::ofstream conoutput;
	std::ofstream rhooutput;
	std::ofstream magoutput;

	// Local box
	std::vector<int> density(L2,0);
	std::vector<int> next_density(L2,0);
	std::vector<int> magnetization(L2,0);
	std::vector<int> next_magnetization(L2,0);

	// Mutex for each box
	std::mutex *density_mutex = new std::mutex[L2];

	// Nearest neighbor of box
	// 0: right, 1: left, 2: up, 3: down
	int **nn = new int*[L2];
	for (i=0;i<L2;++i){
		nn[i] = new int[4];
		nn[i][0] = (i%Lx==Lx-1 ? i+1-Lx : i+1);
		nn[i][1] = (i%Lx==0 ? i-1+Lx : i-1);
		nn[i][2] = (i/Lx==0 ? i-Lx+L2 : i-Lx);
		nn[i][3] = (i/Lx==Ly-1 ? i+Lx-L2 : i+Lx);
	}

	// Initialization
	std::vector<Ptcl> ptcl(N);
	std::vector<Ptcl> next_ptcl(N);
	ptcl.reserve(N);
	next_ptcl.reserve(N);

	if (FIRST==1){
		for (i=0;i<N;++i){
			ptcl[i].pos= L2*uni(gen);
			ptcl[i].spin = (uni(gen)>0.5 ? 1: -1);
			density[ptcl[i].pos] += 1;
			magnetization[ptcl[i].pos] += ptcl[i].spin;
			//printf("%.2f %.2f %.2f %d\n",ptcl[i].x,ptcl[i].y,ptcl[i].angle,bpos);
		}
	} else{
		std::ifstream inputfile;
		inputfile.open(confile);
		inputfile >> t;
		for (i=0;i<N;++i){
			inputfile >> ptcl[i].pos >> ptcl[i].spin;
			density[ptcl[i].pos] += 1;
			magnetization[ptcl[i].pos] += ptcl[i].spin;
		}
		inputfile.close();
		tf += t;
	}

	mag = mag2 = mag4 = 0;

	// Put ptcl into box
	while (t<tf){
#pragma omp parallel for num_threads(NUM_THREADS) default(shared) private(pos,spin,prob_r,prob_l,exponent,prob_flip,prob,right,left,up,down) schedule(auto)
		for (n=0;n<N;++n){
			// Calculate probabilities
			std::default_random_engine& engine = generators[omp_get_thread_num()];
			std::uniform_real_distribution<double> uni(0,1);
			pos = ptcl[n].pos;
			spin = ptcl[n].spin;
			prob_r = 0.5*(1.0+spin*e);
			prob_l = 0.5*(1.0-spin*e);
			exponent = -beta*spin*magnetization[pos]/density[pos];
			prob_flip = exp(exponent)*dt;
			prob = uni(gen);
			//std::cout << n << " " << pos << " " << spin << " " << density[pos] << " " << magnetization[pos]  << " " << prob_flip<< "\n";
			next_ptcl[n].pos = pos;
			next_ptcl[n].spin = spin;
			// Horizontal hopping
			if (prob<prob_move){
				if (uni(gen)<prob_r){
					right = nn[pos][0];
					//std::cou << right << " " << omp_get_thread_num() << "\n";
					density_mutex[right].lock();
					// Update
					next_density[right] += 1;
					next_magnetization[right] += spin;
					density_mutex[right].unlock();
					next_ptcl[n].pos = right;
				} else{
					left = nn[pos][1];
					//std::cout << left << " " << omp_get_thread_num() << "\n";
					density_mutex[left].lock();
					// Update
					next_density[left] += 1;
					next_magnetization[left] += spin;
					density_mutex[left].unlock();
					next_ptcl[n].pos = left;
				}
			} else if (prob<prob_move2){
				if (uni(gen)<0.5){
					up = nn[pos][2];
					//std::cout << up << " " << omp_get_thread_num() << "\n";
					density_mutex[up].lock();
					// Update
					next_density[up] += 1;
					next_magnetization[up] += spin;
					density_mutex[up].unlock();
					next_ptcl[n].pos = up;
				} else{
					down = nn[pos][3];
					density_mutex[down].lock();
					//std::cout << down << " " << omp_get_thread_num() << "\n";
					next_density[down] += 1;
					next_magnetization[down] += spin;
					density_mutex[down].unlock();
					next_ptcl[n].pos = down;
				}
			} else if (prob<prob_move2+prob_flip){
				next_ptcl[n].spin = -spin;
				density_mutex[pos].lock();
				next_density[pos] += 1;
				next_magnetization[pos] -= 2*spin;
				density_mutex[pos].unlock();
			}	else{
				density_mutex[pos].lock();
				next_density[pos] += 1;
				next_magnetization[pos] += spin;
				density_mutex[pos].unlock();
			}
		}
		density.swap(next_density);
		magnetization.swap(next_magnetization);
		std::fill(next_density.begin(), next_density.end(), 0);
		std::fill(next_magnetization.begin(), next_magnetization.end(), 0);
		ptcl.swap(next_ptcl);

		if (FIRST == 0){
			m=0;
			for (int mag_site : magnetization) m += mag_site;
			m = fabs(m)/L2;
			mag += m;
			mag2 += m*m;
			mag4 += m*m*m*m;
		}

		t += dt;

		++cnt1;
		++cnt2;
		// Write observables
		if (cnt1==cnter1){
			// std::cout << t << "\n";
			if (FIRST == 0){
				mag /= cnter1;
				mag2 /= cnter1;
				mag4 /= cnter1;
				obsoutput.open(obsfile,std::ios::app);
				obsoutput << mag << " " << mag2 << " " << mag4 << " "
									<< Lx << " " << Ly << " " << beta << " " << rho << " " << e << "\n";
				mag = mag2 = mag4 = 0;
				obsoutput.close();
			}
			cnt1 = 0;	
		}
		// // Write configuration
		// if (cnt2==cnter2){
		// 	conoutput.open(confile);
		// 	conoutput << t << "\n";
		// 	for (i=0;i<N;++i) conoutput << ptcl[i].pos << " " << ptcl[i].spin << "\n";
		// 	conoutput.close();
		// 	cnt2 = 0;	
		// }
	}

	conoutput.open(confile);
	conoutput << t << "\n";
	for (i=0;i<N;++i) conoutput << ptcl[i].pos << " " << ptcl[i].spin << "\n";
	conoutput.close();
	
	rhooutput.open(rhofile);
	rhooutput << t << " ";
	for (i=0;i<L2;++i){
		rhooutput << density[i] << " ";
	}

	magoutput.open(magfile);
	magoutput << t << " ";
	for (i=0;i<L2;++i){
		magoutput << magnetization[i] << " ";
	}

	for (i=0;i<L2;++i) delete[] nn[i];
	delete[] nn;

	return 0;
}


