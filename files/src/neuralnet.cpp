#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "neuralnet.h"

using namespace std;
using namespace rf::ai;

static inline float sigmoid(float x)
{
	return 1.0f  / (1.0f + expf(-x));
}

Neuron::Neuron()
	:
	signal_(0.0f),
	bias_(1.0f)
{
}

Neuron::~Neuron()
{
}
			
void Neuron::add(shared_ptr<Neuron> in_neuron)
{
	auto s = make_shared<synapse>(); 
	in_neuron->out_.push_back(s);
	this->in_.push_back(s);
	
	s->delta = 0;
	s->weight = ((float)rand())/RAND_MAX/100.0f;
	s->dweight = 0;
}

float Neuron::get_signal()
{
	return signal_;
}

void Neuron::set_signal(float signal)
{
	signal_ = signal;
}

void Neuron::update_signal()
{
	if (in_.size() != 0)
	{
		float p = bias_;
		for (auto v : in_)
		{
			v->weight += v->dweight;
			v->prev_dweight = v->dweight;
			v->dweight = 0.0f;
			p +=  v->weight * v->signal;
		}
		signal_ = sigmoid(p);
	}
	
	for (auto v : out_)
	{
		v->signal = signal_;
	}
}

void Neuron::kill()
{
	printf("invalid function\n");
}

static const float eta = 1.0f; // (0���Ł�1)
static const float alpha = 0.1f; // �����`��

void Neuron::backpropagation(float teaching_signal)
{
	// �ŋ}�~���@
	// E=1/2��(ti-oi)^2
	// ��w = -�� * ��E/��w �����߂�
	// ���ԑw���o�͑w�i���̑w�j����
	// �o�͑w�̂���j���[����j�ɂ��āAk�Ԗڂ̓��͉׏d���X�V����B
	// O:�o�́AU:�l�b�g�l(����*�׏d�̍��v�l)�Ƃ��A
	// E��k�Ԗڂ̉׏d�ō����֐��̕Δ���(!k�ԖڈȊO���Œ�l�Ƃ��Ĕ���!)
	//    ��w = -�� * ��E/��O*��O/��U*��U/��wk 
	//       ��E/��O = -(t-o)��
	//       ��O/��U = o(1-o)���� // �V�O���C�h�֐��̔���
	//       ��U/��wk=��((i1*w1)+(i2*w2)...(in*wn))/��wk = ik��
	//       �����A��(t-o)o(1-o)ik
	// �B��w
	// H:�B��w�̏o�́AT�B��w�̃l�b�g�l
	//    ��w = -�� * ��(��E/��O*��O/��U*��U/��H)*��H/��T*��T/��wk
	//       ��E/��O*��O/��U�́��Ɓ����o�͑w�̌v�Z�Ń�j=(t-o)o(1-o)�Ƃ��Ċo���Ă���
	//       ��H/��T�́�O/��U�Ƃ��Ȃ��v�Z�A��T/��wk����U/��wk �Ɠ����v�Z
	//       �����A�Ń�(��j*wj)*o*(1-o)*ik
	if (out_.size() == 0)
	{
		// �o�͌덷 delta�����߂�
		float gradient = signal_ * (1.0f - signal_);
		float delta = -(teaching_signal - signal_) * gradient;
		
		// �o�C�A�X���X�V
		bias_ += -(eta * delta * 0.1f); // 0.1�ɂ��Ȃ��ƌ덷�����̂܂ܑ������H
		
		// weight���X�V
		for (auto v : in_)
		{
			v->delta = delta;
			
			// ���͂��o�͌덷�ɗ^����e��
			v->dweight -= eta * delta * v->signal;

			// �����W��
			v->dweight += alpha * v->prev_dweight;
		}
	}
	else
	{
		// �o�͌덷 delta�̑��a�����߂�
		float gradient = signal_ * (1.0f - signal_);
		float delta = 0;
		for (auto v : out_)
		{
			delta += v->delta * v->weight;
		}
		delta *= gradient;
		
		// �o�C�A�X���X�V
		bias_ += -(eta * delta * 0.1f); // 0.1�ɂ��Ȃ��ƌ덷�����̂܂ܑ������H
		
		// weight���X�V
		for (auto v : in_)
		{
			v->delta = delta;
			
			// ���͂��o�͌덷�ɗ^����e��
			v->dweight -= eta * delta * v->signal;

			// �����W��
			v->dweight += alpha * v->prev_dweight;
		}
	}
}

