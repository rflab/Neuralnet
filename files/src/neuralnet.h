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
		// ニューロン一個
		class Neuron
		{
		public:
			struct synapse{
				float signal; // 今現在の出力誤差信号
				float delta; // 今現在の出力誤差信号
				float weight; // 今現在のweight
				float dweight; // 次に信号が通った時に使うweightの補正量
				float prev_dweight; // 次に信号が通った時に使うweight
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

