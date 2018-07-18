#include <stdio.h>
#include "MLP.h"

int main()
{
	int nodes[] = { 40, 40 };
	MLP* mlp = new MLP(0.1, 10, 2, nodes, "UCI.txt", "UCI_test.txt",10000);
	mlp->Train();
	
	return 0;
}