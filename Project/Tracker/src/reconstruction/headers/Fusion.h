#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include "../../concurency/sources/Buffer.cpp"
#include "../../concurency/sources/Consumer.cpp"
#include "FusionBase.h"

using namespace std; 

/**
 * Volumetric m_fusion class
 */
class Fusion final : public FusionBase  
{
public:
	Fusion(SystemParameters camera_parameters);

	~Fusion();

	void consume() override;

	void produce(std::shared_ptr<PointCloud> cloud) override;

	void integrate(std::shared_ptr<PointCloud> cloud) override;

	void save(string name) override;

	void processMesh(Mesh& mesh) override;

	void wait() const override;

private:
	void initialize();

	void stopConsumers();

	float getWeight(float depth, float max) const;

	std::vector<std::thread> m_consumer_threads;
	std::vector<Consumer<std::shared_ptr<PointCloud>>*> m_consumers;
	Buffer<std::shared_ptr<PointCloud>>* m_buffer;
	std::mutex m_mutex;
	const int NUMBER_OF_CONSUMERS = 5;
};

#endif //TRACKER_LIB_FUSION_H
