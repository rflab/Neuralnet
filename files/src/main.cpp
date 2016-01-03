#include <cstdio>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "neuralnet.h"

using namespace std;

void test()
{
	vector<shared_ptr<rf::ai::Neuron> > neuralnet;
	for (int i=0; i<10; i++)
	{
		neuralnet.push_back(make_shared<rf::ai::Neuron>());
	}

	neuralnet[0]->add(neuralnet[2]);
	neuralnet[0]->add(neuralnet[3]);
	neuralnet[0]->add(neuralnet[4]);
	neuralnet[0]->add(neuralnet[5]);
	
	neuralnet[1]->add(neuralnet[2]);
	neuralnet[1]->add(neuralnet[3]);
	neuralnet[1]->add(neuralnet[4]);
	neuralnet[1]->add(neuralnet[5]);

	neuralnet[2]->add(neuralnet[6]);
	neuralnet[2]->add(neuralnet[7]);
	neuralnet[3]->add(neuralnet[6]);
	neuralnet[3]->add(neuralnet[7]);
	neuralnet[4]->add(neuralnet[6]);
	neuralnet[4]->add(neuralnet[7]);
	neuralnet[5]->add(neuralnet[6]);
	neuralnet[5]->add(neuralnet[7]);
	
	//neuralnet[2]->add(neuralnet[8]);
	//neuralnet[2]->add(neuralnet[9]);
	//neuralnet[3]->add(neuralnet[8]);
	//neuralnet[3]->add(neuralnet[9]);
    //
	//neuralnet[8]->add(neuralnet[6]);
	//neuralnet[8]->add(neuralnet[7]);
	//neuralnet[9]->add(neuralnet[6]);
	//neuralnet[9]->add(neuralnet[7]);
	

	float dummy = 0.0f;
	for (int i = 0; i<1000; ++i)
	{
		printf("==============================\n");
		neuralnet[6]->set_signal(0.0f);
		neuralnet[7]->set_signal(0.0f);
		
		neuralnet[6]->update_signal();
		neuralnet[7]->update_signal();
		neuralnet[8]->update_signal();
		neuralnet[9]->update_signal();
		neuralnet[2]->update_signal();
		neuralnet[3]->update_signal();
		neuralnet[4]->update_signal();
		neuralnet[5]->update_signal();
		neuralnet[0]->update_signal();
		neuralnet[1]->update_signal();
		
		neuralnet[0]->backpropagation(0.0f);
		neuralnet[1]->backpropagation(0.0f);
		neuralnet[2]->backpropagation(dummy);
		neuralnet[3]->backpropagation(dummy);
		neuralnet[4]->backpropagation(dummy);
		neuralnet[5]->backpropagation(dummy);
		neuralnet[8]->backpropagation(dummy);
		neuralnet[9]->backpropagation(dummy);
		
		std::cout << "0 "<< neuralnet[0]->get_signal() << endl;
		std::cout << "0 "<< neuralnet[1]->get_signal() << endl;
		std::cout << std::endl;
        
		printf("-----------\n");
		neuralnet[6]->set_signal(1.0f);
		neuralnet[7]->set_signal(0.0f);
		
		neuralnet[6]->update_signal();
		neuralnet[7]->update_signal();
		neuralnet[8]->update_signal();
		neuralnet[9]->update_signal();
		neuralnet[2]->update_signal();
		neuralnet[3]->update_signal();
		neuralnet[4]->update_signal();
		neuralnet[5]->update_signal();
		neuralnet[0]->update_signal();
		neuralnet[1]->update_signal();
		
		neuralnet[0]->backpropagation(1.0f);
		neuralnet[1]->backpropagation(1.0f);
		neuralnet[2]->backpropagation(dummy);
		neuralnet[3]->backpropagation(dummy);
		neuralnet[4]->backpropagation(dummy);
		neuralnet[5]->backpropagation(dummy);
		neuralnet[8]->backpropagation(dummy);
		neuralnet[9]->backpropagation(dummy);
		
		std::cout << "1 "<< neuralnet[0]->get_signal() << endl;
		std::cout << "1 "<< neuralnet[1]->get_signal() << endl;
		std::cout << std::endl;
        
		printf("-----------\n");
		neuralnet[6]->set_signal(0.0f);
		neuralnet[7]->set_signal(1.0f);
		
		neuralnet[6]->update_signal();
		neuralnet[7]->update_signal();
		neuralnet[8]->update_signal();
		neuralnet[9]->update_signal();
		neuralnet[2]->update_signal();
		neuralnet[3]->update_signal();
		neuralnet[4]->update_signal();
		neuralnet[5]->update_signal();
		neuralnet[0]->update_signal();
		neuralnet[1]->update_signal();
		
		neuralnet[0]->backpropagation(1.0f);
		neuralnet[1]->backpropagation(1.0f);
		neuralnet[2]->backpropagation(dummy);
		neuralnet[3]->backpropagation(dummy);
		neuralnet[4]->backpropagation(dummy);
		neuralnet[5]->backpropagation(dummy);
		neuralnet[8]->backpropagation(dummy);
		neuralnet[9]->backpropagation(dummy);
		
		std::cout << "1 "<< neuralnet[0]->get_signal() << endl;
		std::cout << "1 "<< neuralnet[1]->get_signal() << endl;
		std::cout << std::endl;
        
		printf("-----------\n");
		neuralnet[6]->set_signal(1.0f);
		neuralnet[7]->set_signal(1.0f);
		
		neuralnet[6]->update_signal();
		neuralnet[7]->update_signal();
		neuralnet[8]->update_signal();
		neuralnet[9]->update_signal();
		neuralnet[2]->update_signal();
		neuralnet[3]->update_signal();
		neuralnet[4]->update_signal();
		neuralnet[5]->update_signal();
		neuralnet[0]->update_signal();
		neuralnet[1]->update_signal();
		
		neuralnet[0]->backpropagation(0.0f);
		neuralnet[1]->backpropagation(1.0f);
		neuralnet[2]->backpropagation(dummy);
		neuralnet[3]->backpropagation(dummy);
		neuralnet[4]->backpropagation(dummy);
		neuralnet[5]->backpropagation(dummy);
		neuralnet[8]->backpropagation(dummy);
		neuralnet[9]->backpropagation(dummy);
		
		std::cout << "0 "<< neuralnet[0]->get_signal() << endl;
		std::cout << "1 "<< neuralnet[1]->get_signal() << endl;
		std::cout << std::endl;
	}
}

int main(int argc, char** argv)
{
	printf("argc%d, argv%ss\n", argc, argv[0]);	
	test();
	return 0;
}



