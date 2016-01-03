#ifndef _RF_NEURALNET_
#define _RF_NEURALNET_

#include <memory>
#include <list>
#include <set>
#include <tuple>

namespace rf
{
	namespace ai
	{
		// �j���[�������
		class Neuron
		{
		public:
			struct synapse{
				float signal; // �����݂̏o�͌덷�M��
				float delta; // �����݂̏o�͌덷�M��
				float weight; // �����݂�weight
				float dweight; // ���ɐM�����ʂ������Ɏg��weight�̕␳��
				float prev_dweight; // ���ɐM�����ʂ������Ɏg��weight
			};
			
			std::list<std::shared_ptr<synapse> > out_;
			std::list<std::shared_ptr<synapse> > in_;
			float signal_;
			float bias_;
			
		protected:
			
		public:
			Neuron();
			virtual ~Neuron();
			
			void add(std::shared_ptr<Neuron> in);
			void remove(std::shared_ptr<Neuron> neuron);
			void kill();
			void backpropagation(float teaching_signal);
			float get_signal();
			void set_signal(float signal);
			void update_signal();
		};
	}
}

#endif

