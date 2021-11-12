#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<string.h>
#include<time.h>

#define INPUT 			14
#define MAXIT			100
#define ALPHA 			0.25
#define L_LINES 		123
#define V_LINES 		34
#define T_LINES 		21
#define NEURON 			3
#define NEWLINE 		-38
#define DOT 			-2
#define L_LOCATION		"EXP1/learning.h"
#define V_LOCATION		"EXP1/validation.h"
#define T_LOCATION		"EXP1/test.h"


// Declaração das funções
double sigma(double *dread,double *weight);
void softmax(double *y);
void init_weight(double x[][INPUT]);
void update_weight(double *dread,double e,double *weight);
void update_bias(double *weight,double e);
void init_e(double *e);
void input(double dread[][INPUT],int *d);
void training(double alpha,double dread[][INPUT],double vdread[][INPUT],double d[][NEURON],double d_eval[][NEURON],double weight[][INPUT]);
void test(double dread[][INPUT],double weight[][INPUT],double d_eval[][NEURON],int lines);
void readfile(double dread[][INPUT],double d[][NEURON],int lines,char *location);
void normalize(double dread[][INPUT]);
int findmax(double *y);
void zscore(double dread[][INPUT],int lines);
double getmean(double dread[][INPUT],int col,int lines);
double getstddev(double dread[][INPUT],int col,double mean,int lines);
double evaluation(double vdread[][INPUT],double weight[][INPUT],double d_eval[][NEURON],double error_t);

FILE *ValError;
FILE *LearnError;

int main(void){
	ValError = fopen("ValError.txt","w");
	LearnError = fopen("LearnError.txt","w");
	// Input
	double dread[L_LINES][INPUT];
	double vdread[V_LINES][INPUT];
	double tdread[T_LINES][INPUT];
	// Vetor de pesos
	double weight[NEURON][INPUT];
	// Saída desejada
	double d[L_LINES][NEURON];
	double d_eval[V_LINES][NEURON];
	double d_test[T_LINES][NEURON];
	double tinput[NEURON][INPUT];

	readfile(dread,d,L_LINES,L_LOCATION);
	readfile(vdread,d_eval,V_LINES,V_LOCATION);
	readfile(tdread,d_test,T_LINES,T_LOCATION);
	zscore(dread,L_LINES);
	zscore(vdread,V_LINES);
	zscore(tdread,T_LINES);
	
	
	training(ALPHA,dread,vdread,d,d_eval,weight);


	printf("Imprimindo matrizes de confusão na seguinte ordem:"
		" Teste, Validação e Aprendizado\n");
	test(tdread,weight,d_test,T_LINES);
	test(vdread,weight,d_eval,V_LINES);
    test(dread,weight,d,L_LINES);


    fclose(ValError);
    fclose(LearnError);
}

void displaymatrix(double d_eval[][NEURON],int lines){


	for(int i=0;i<lines;i++){
		printf("\n");
		for(int k=0;k<INPUT;k++){
				printf("%lf ",d_eval[i][k]);
		}
	}
	printf("\n");

}

void test(double tdread[][INPUT],double weight[][INPUT],double d_test[][NEURON],int lines){

	double cmatrix[NEURON][NEURON];
	double y[NEURON];
	int index;
	memset(cmatrix,0,sizeof(cmatrix));
	for(int i=0;i<lines;i++){
		for(int k=0;k<NEURON;k++){
				y[k] = sigma(tdread[i],weight[k]);
		}
		softmax(y);

		// Visualizado somente no debug
		/*
		for(int i=0;i<NEURON;i++)
				printf(" %lf",y[i]);

		printf("\n");
		*/
	
		
		cmatrix[findmax(y)][findmax(d_test[i])] += 1;
	}

	for(int i=0;i<NEURON;i++){
		printf("\n");
		for(int k=0;k<NEURON;k++){
				printf("%lf ",cmatrix[i][k]);
		}
	}
	printf("\n");
}

int findmax(double *y){

	int index=0;
	for(int i=1;i<NEURON;i++){
		if(y[index] < y[i]){
			index=i;
		}
	}
	return index;
}


void input(double dread[][INPUT],int *d){
	printf("Inputs\n");

	for(int i=0;i<L_LINES;i++){
		for(int j=0;j<INPUT;j++){
			printf("Linha : %d Coluna :%d\n",i,j);
			scanf("%lf",&dread[i][j]);
		}
	}

	printf("Saída\n");
	for(int i=0;i<L_LINES;i++){
		printf("Coluna :%d\n",i);
		scanf("%d",&d[i]);
	}

}



void training(double alpha,double dread[][INPUT],double vdread[][INPUT],double d[][NEURON],double d_eval[][NEURON],double weight[][INPUT]){

	// Bias setado a zero
	int b=0;
	int t=0;
	// Variável Erro
	double ERROR=1;
	int i,k,eval,eval_exit=0;
	double eval_before=0,eval_current=0;
	double e[NEURON];
	double y[NEURON];
	double sum;
	// Pesos recebem zero
	init_weight(weight);
	// Vetor erro com valor zero
	init_e(e);
	while((t < MAXIT) && (ERROR > 0) && (eval_exit <= 5)){
		ERROR=0;
		for(i=0;i<L_LINES;i++){
			for(k=0;k<NEURON;k++){
				y[k] = sigma(dread[i],weight[k]);
				//fprintf(stderr,"Saída sigma %lf\n",y[k]);
			}
			
			softmax(y);
			for(k=0;k<NEURON;k++){
				//fprintf(stderr,"Camada %d\n",k);
				e[k] = d[i][k] - y[k];
				update_weight(dread[i],e[k],weight[k]);
				ERROR+=pow(e[k],2);
			}
		}
		eval_current = evaluation(vdread,weight,d_eval,eval_before);
		fprintf(LearnError,"%d %lf\n",t,ERROR);
		fprintf(ValError,"%d %lf\n",t,eval_current);
		if(eval_current > eval_before)
			eval_exit += 1;
		else
			eval_exit = 0;
		t++;
		eval_before = eval_current;
	}
	

}

double evaluation(double vdread[][INPUT],double weight[][INPUT],double d_eval[][NEURON],double error_t){
	double e[NEURON],y[NEURON],error_v=0;
	int i,k;
		for(i=0;i<V_LINES;i++){
			for(k=0;k<NEURON;k++){
					y[k] = sigma(vdread[i],weight[k]);
					//fprintf(stderr,"Saída sigma %lf\n",y[k]);
				}
				
			softmax(y);
			for(k=0;k<NEURON;k++){
					//fprintf(stderr,"Camada %d\n",k);
					e[k] = d_eval[i][k] - y[k];
					error_v+=pow(e[k],2);
			}
	  }
	  return error_v;
}

void update_bias(double *weight,double e){
	  weight[0] = weight[0] + ALPHA*e;
}

void update_weight(double *dread,double e,double *weight){
	//fprintf(stderr, " Atualizando pesos\n");
	int i=0;
	for(i=0;i<INPUT;i++){
		weight[i] = weight[i] + ALPHA*e*dread[i];
		//fprintf(stderr,"  Weight[%d]:%lf x[%d]:%lf E %lf\n",i,weight[i],i,dread[i],e);
	}
	
}

void softmax(double *y){
	double sum=0;
	int i,k;

	for(k=0;k<NEURON;k++)
		sum+=exp(y[k]); 
	
	for(i=0;i<NEURON;i++){
		y[i]= exp(y[i])/sum;
	}

}

double sigma(double *dread,double *weight){

	double acc=0;
	for(int i=0;i<INPUT;i++){
		acc+= dread[i]*weight[i];
	}

	return acc;
	
}

void init_weight(double x[][INPUT]){
	// Pesos começam com zero
	for(int i=0;i<NEURON;i++){
		for(int j=0;j<INPUT;j++)
		 	x[i][j]=0;
	}

}

void init_e(double *x){
	int i;
	// Iniciar com zero
	for(int k=0;k<NEURON;k++)
		    x[k]=0;
    
}

void readfile(double xinput[][INPUT],double d[][NEURON],int f_lines,char *location){
	FILE *fp = fopen(location,"r");
	int integer[10];
	int dot[10];
	int buff=0,i=0,i_dot=0;
	int k;
	double acc=0;

	for(int lines=0;lines<f_lines;lines++){
			xinput[lines][0] = 1;
			buff=getc(fp)-48;
			switch(buff){
				case 1:
						d[lines][0] = 1;
						d[lines][1] = 0;
						d[lines][2] = 0;
				break;
				case 2:
						d[lines][0] = 0;
						d[lines][1] = 1;
						d[lines][2] = 0;
				break;
				case 3:
						d[lines][0] = 0;
						d[lines][1] = 0;
						d[lines][2] = 1;

			}
			buff=getc(fp)-48;
			for(int attributes=1;attributes<INPUT;attributes++){
					i=0;
					i_dot=0;
					acc=0;
					buff=getc(fp)-48;
					while(buff != NEWLINE && buff != DOT){
					  integer[i++] = buff;
					  buff=getc(fp)-48;
					}
					if(buff==DOT){
					  buff=getc(fp)-48;
					   while(buff != NEWLINE){
					      dot[i_dot++] = buff;
					      buff=getc(fp)-48;
						}		
					}
					for(k=0;k<i;k++)
						acc+=integer[k]*pow(10,(i-1)-k);

					for(k=0;k<i_dot;k++)
						acc+=dot[k]/pow(10,k+1);

					xinput[lines][attributes] = acc;

		}
	}
	fclose(fp);
}




// Realiza o cálculo de zscore
void zscore(double dread[][INPUT],int lines){

	double mean;
	double stddev;
	int i,j;


	//Pulando o bias
	for(j=1;j<INPUT;j++){
		mean	= getmean(dread,j,lines);
		stddev	= getstddev(dread,j,mean,lines);
		//sleep(10);
		for(i=0;i<lines;i++){
			dread[i][j] = (dread[i][j] - mean)/ (stddev);
		}
	}
}

// Média de cada atributo
double getmean(double dread[][INPUT],int col,int lines){
	double med=0;
	int i;
	for(i=0;i<lines;i++)
		med+=dread[i][col];

	return med/lines;
} 

// Desvio padrão de cada atributo
double getstddev(double dread[][INPUT],int col,double mean,int lines){
	double stddev=0;
	int i;
	for(i=0;i<lines;i++){
		//fprintf(stderr,"%lf e %lf\n",(dread[i][col] - mean),mean);
		stddev+=pow(dread[i][col] - mean,2);
	}

	stddev=sqrt(stddev/lines);

	return stddev;
} 
