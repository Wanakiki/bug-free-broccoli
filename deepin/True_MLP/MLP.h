#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
using namespace std;
float sample_from_gaussian(float miu, float sigma){
	static float V1, V2, S;
	static int phase = 0;
	float X;
	float gaussian_output;
	if (phase == 0){
		do{
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);
		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);
	phase = 1 - phase;
	gaussian_output = X * sigma + miu;
	return gaussian_output;
}

//sigmoid
float sigmoid(float x){
	return  1 / (1 + exp(-x));
}

//derivative of sigmoid
float dsigmoid(float y){
	return y*(1-y);
}
class Layer
{
public:
	int layer_id;
	int node_num;
	float* b;
	float* db;
	float* u;
	float* du;
	float* y;	//y=f(u)=f(wx+b)
	float* dy;
	float** w;//pre_nodes*current_nodes
	float** dw;
	Layer* pre_layer;
	Layer* next_layer;
	Layer()
	{
	}
	void Set_nodes(int n)
	{
		node_num = n;
	}
	void Init_params()
	{
		w = new float*[pre_layer->node_num];
		dw = new float*[pre_layer->node_num];
		for (int i = 0; i < pre_layer->node_num; i++)
		{
			w[i] = new float[node_num];
			dw[i] = new float[node_num];
			for (int j = 0; j < node_num; j++)
			{
				w[i][j] = sample_from_gaussian(0, 0.01);
				dw[i][j] =0;
			}
		}
		u = new float[node_num];
		du = new float[node_num];
		y = new float[node_num];
		dy = new float[node_num];
		b = new float[node_num];
		db = new float[node_num];
		for (int i = 0; i < node_num; i++)
		{
			u[i] = 0;
			du[i] = 0;
			y[i] = 0;
			dy[i] = 0;
			b[i] = 0;
			db[i] = 0;
		}
	}
	void Active()
	{
		for (int i = 0; i < node_num; i++)
			y[i] = sigmoid(u[i]);
	}
};
class Data_Layer: public Layer
{
public:
	int data_num;
	int* label;	// M or B
	int label_count;
	float** data;
	void Read_data(const char* data_path)
	{
		string ReadLine;
		char* ContLine;
		char tempchar;
		vector<char> label_lists;
		bool d_flag = false;
		//getline(fin,ReadLine);
		bool initial_flag = false;
		int label_id = 0;
		fstream fin(data_path);
		while (getline(fin, ReadLine))
		{
			node_num = 0;
			for (int i = 0; i<ReadLine.length(); i++)
			{
				if (ReadLine[i] == '\t')
					node_num++;
			}	//发现一个特征  增加一个节点
			bool flag = false;
			for (int i = 0; i < label_lists.size(); i++)
			{
				if (label_lists[i] == ReadLine[0])
				{
					flag = true;
					break;
				}
			}	//判断labe_list是否含有首个元素
			if (!flag)
				label_lists.push_back(ReadLine[0]);
			data_num++;
		}	//将M B添加进label_lists
		//仅有两个例子的情况可以直接添加，但是为了解决更多的问题应该考虑这样去写
		fin.close();
		label_count = label_lists.size();
		label = new int[data_num];
		data = new float*[data_num];
		y = new float[node_num];
		for (int i = 0; i < data_num; i++)
		{
			data[i] = new float[node_num];
			for (int j = 0; j < node_num; j++)
				data[i][j] = 0;
		}
		FILE* fp;
		fp = fopen(data_path, "r");
		for (int i = 0; i < data_num; i++)
		{
			fscanf(fp, "%c\t", &tempchar);
			if (tempchar == 'M')
				label[i] = 0;
			else
				label[i] = 1;
			for (int j = 1; j < node_num+1; j++)
			{
				fscanf(fp, "%f\t", &data[i][j-1]);
			}

		}
	}

};
class Out_Layer : public Layer
{
	public:
	void Active()
	{
		float sum = 0;
		for (int i = 0; i < node_num; i++)
		{q
			y[i] = exp(u[i]);
			sum += exp(u[i]);
		}
		for (int i = 0; i < node_num; i++)
			y[i] /=sum;
	}
};
class MLP
{
public:
	int layer_num;
	Data_Layer* train_layer;
	Data_Layer* test_layer;
	Out_Layer* out_layer;
	Layer* layers;
	float learning_rate;
	int batch_size;
	int epoches;
	float train_acc;
	float test_acc;
	float error;
	//MLP::MLP(float lr,int bs,int layer_n,int node_nums[],const char train_path[],const char test_path[],int ep)
	MLP(float lr,int bs,int layer_n,int node_nums[],const char train_path[],const char test_path[],int ep)
	{
		learning_rate = lr;
		batch_size = bs;
		layer_num = layer_n;
		epoches = ep;
		train_layer = new Data_Layer();
		test_layer = new Data_Layer();
		train_layer->Read_data(train_path);
		test_layer->Read_data(test_path);
		out_layer = new Out_Layer();
		out_layer->Set_nodes(train_layer->label_count);
		layers = new Layer[layer_num]();
		train_layer->next_layer = &layers[0];
		test_layer->next_layer = &layers[0];
		for (int i = 0; i < layer_num; i++)
		{
			layers[i].node_num = node_nums[i];
			if (i == 0)
			{
				layers[i].pre_layer = (Layer*)train_layer;
			}
			else
			{
				layers[i].pre_layer = &layers[i - 1];
			}
			if (i != layer_num - 1)
			{
				layers[i].next_layer = &layers[i + 1];
			}
			else
			{
				layers[i].next_layer = (Layer*)out_layer;
			}
			layers[i].Init_params();
		}
		out_layer->pre_layer = &layers[layer_num - 1];
		out_layer->next_layer = NULL;
		out_layer->Init_params();
	}
	int Forward(Data_Layer* data_layer,int data_id)
	{
		int max_id = 0;
		float max_value = 0;
			for (int j = 0; j < data_layer->node_num; j++)
				data_layer->y[j] = data_layer->data[data_id][j];
			for (int j = 0; j < layer_num; j++)
			{

				for (int k = 0; k < layers[j].node_num; k++)
				{
					layers[j].u[k] = 0;
					for (int m = 0; m < layers[j].pre_layer->node_num; m++)
					{
						layers[j].u[k] += layers[j].w[m][k] * layers[j].pre_layer->y[m];
					}
					layers[j].u[k] += layers[j].b[k];
				}
				layers[j].Active();
			}


			for (int j = 0; j < out_layer->node_num; j++)
			{
				out_layer->u[j] = 0;
				for (int k = 0; k < out_layer->pre_layer->node_num; k++)
				{
					out_layer->u[j] += out_layer->pre_layer->y[k] * out_layer->w[k][j];
				}

			}
			out_layer->Active();
			for (int j = 0; j < out_layer->node_num;
				 j++){
					 printf("node_num:%d %d , %d, %d, %d\n",out_layer->node_num, j,max_value,max_id,out_layer->y[j]);
				if (max_value < out_layer->y[j])
				{
					max_value = out_layer->y[j];
					max_id = j;
				}
			}
			printf("max_id is:%d, data_val is:%d\n", max_id,data_layer->label[data_id]);
			if (max_id == data_layer->label[data_id])
				return 1;
			else
				return 0;

	}
	void Backward(Data_Layer* data_layer,int data_id)
	{

			for (int j = 0; j < out_layer->node_num; j++)
			{
				out_layer->dy[j] = 0;
				if (j == data_layer->label[data_id])
					out_layer->dy[j] += out_layer->y[j] - 1;
				else
					out_layer->dy[j] += out_layer->y[j] - 0;
				out_layer->du[j] = out_layer->dy[j];
				error += out_layer->dy[j] * out_layer->dy[j] / out_layer->node_num;
				for (int m = 0; m < out_layer->pre_layer->node_num; m++)
				{
					out_layer->dw[m][j] += out_layer->pre_layer->y[m] * out_layer->du[j];
				}
			}
			for (int j = layer_num - 1; j >= 0; j--)
			{
				for (int k = 0; k < layers[j].node_num; k++)
				{
					layers[j].dy[k] = 0;
					for (int m = 0; m < layers[j].next_layer->node_num; m++)
						layers[j].dy[k] += layers[j].next_layer->du[m]*layers[j].next_layer->w[k][m];
					layers[j].du[k] = layers[j].dy[k] * dsigmoid(layers[j].y[k]);
					layers[j].db[k] += layers[j].du[k];
					for (int m = 0; m < layers[j].pre_layer->node_num; m++)
					{
						layers[j].dw[m][k] += layers[j].pre_layer->y[m] * layers[j].du[k];
					}
				}
			}
	}
	void Update()
	{
		for (int k = 0; k < out_layer->node_num; k++)
		{
			for (int j = 0; j < out_layer->pre_layer->node_num; j++)
			{
				out_layer->w[j][k] -= learning_rate*out_layer->dw[j][k] / batch_size;
				out_layer->dw[j][k] = 0;
			}
			out_layer->du[k] = 0;
			out_layer->dy[k] = 0;
		}
		for (int i = 0; i < layer_num; i++)
		{
			for (int k = 0; k < layers[i].node_num; k++)
			{
				for (int j = 0; j < layers[i].pre_layer->node_num; j++)
				{
					layers[i].w[j][k] -= learning_rate*layers[i].dw[j][k] / batch_size;
					layers[i].dw[j][k] = 0;
				}
				layers[i].b[k] -= learning_rate*layers[i].db[k] / batch_size;
				layers[i].db[k] = 0;
				layers[i].du[k] = 0;
				layers[i].dy[k] = 0;
			}
		}

	}
	void Train()
	{

		for (int i = 0; i < epoches; i++)
		{
			int batch_count = 0;
			train_acc = 0;
			test_acc = 0;
			error = 0;
			layers[0].pre_layer = train_layer;
			for (int j = 0; j < train_layer->data_num; j++)
			{
				Forward(train_layer, j);
				Backward(train_layer,j);
				batch_count++;
				if (batch_count >= batch_size)
				{
					batch_count = 0;
					Update();
				}
			}
			for (int j = 0; j < train_layer->data_num; j++)
				train_acc += Forward(train_layer, j);
			layers[0].pre_layer = test_layer;
			for (int j = 0; j < test_layer->data_num;j++)
				test_acc += Forward(test_layer, j);
			printf("epoch:%d\ttrain_acc:%f\terror:\%f\ttest_acc:%f\n", i, train_acc / train_layer->data_num,error, test_acc / test_layer->data_num);
		}
	}
};
