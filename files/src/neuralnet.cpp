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

static const float eta = 1.0f; // (0＜η≦1)
static const float alpha = 0.1f; // 慣性形数

void Neuron::backpropagation(float teaching_signal)
{
	// 最急降下法
	// E=1/2Σ(ti-oi)^2
	// Δw = -η * ∂E/∂w を求める
	// 中間層→出力層（次の層）結合
	// 出力層のあるニューロンjについて、k番目の入力荷重を更新する。
	// O:出力、U:ネット値(入力*荷重の合計値)とし、
	// Eをk番目の荷重で合成関数の偏微分(!k番目以外を固定値として微分!)
	//    Δw = -η * ∂E/∂O*∂O/∂U*∂U/∂wk 
	//       ∂E/∂O = -(t-o)★
	//       ∂O/∂U = o(1-o)★★ // シグモイド関数の微分
	//       ∂U/∂wk=∂((i1*w1)+(i2*w2)...(in*wn))/∂wk = ikで
	//       即ち、η(t-o)o(1-o)ik
	// 隠れ層
	// H:隠れ層の出力、T隠れ層のネット値
	//    Δw = -η * Σ(∂E/∂O*∂O/∂U*∂U/∂H)*∂H/∂T*∂T/∂wk
	//       ∂E/∂O*∂O/∂Uは★と★★出力層の計算でδj=(t-o)o(1-o)として覚えておく
	//       ∂H/∂Tは∂O/∂Uとおなじ計算、∂T/∂wkも∂U/∂wk と同じ計算
	//       即ち、ηΣ(δj*wj)*o*(1-o)*ik
	if (out_.size() == 0)
	{
		// 出力誤差 deltaを求める
		float gradient = signal_ * (1.0f - signal_);
		float delta = -(teaching_signal - signal_) * gradient;
		
		// バイアスを更新
		bias_ += -(eta * delta * 0.1f); // 0.1にしないと誤差がそのまま足される？
		
		// weightを更新
		for (auto v : in_)
		{
			v->delta = delta;
			
			// 入力が出力誤差に与える影響
			v->dweight -= eta * delta * v->signal;

			// 慣性係数
			v->dweight += alpha * v->prev_dweight;
		}
	}
	else
	{
		// 出力誤差 deltaの総和を求める
		float gradient = signal_ * (1.0f - signal_);
		float delta = 0;
		for (auto v : out_)
		{
			delta += v->delta * v->weight;
		}
		delta *= gradient;
		
		// バイアスを更新
		bias_ += -(eta * delta * 0.1f); // 0.1にしないと誤差がそのまま足される？
		
		// weightを更新
		for (auto v : in_)
		{
			v->delta = delta;
			
			// 入力が出力誤差に与える影響
			v->dweight -= eta * delta * v->signal;

			// 慣性係数
			v->dweight += alpha * v->prev_dweight;
		}
	}
}

