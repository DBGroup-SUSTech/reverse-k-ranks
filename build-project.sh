mkdir build
mkdir result
mkdir result/attribution
mkdir result/laptop
mkdir result/plot_performance
mkdir result/rank
mkdir result/single_query_performance
mkdir result/vis_performance
#mkdir index
#mkdir index/memory_index
#mkdir index/qrs_to_sample_index
#mkdir index/query_distribution
#mkdir index/svd_index

sudo apt-get install libopenblas-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libarmadillo-dev
# install spdlog in the github, should compile by it self
# current spdlog is 1.10
# upgrade the gcc / g++ version to 9.4.0
# install boost-1.80
# the cuda version is 11.8, should install thrust from the github
