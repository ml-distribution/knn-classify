package zx.soft.mahout.knn;

import java.util.List;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.ChineseRestaurant;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.Sampler;

import com.google.common.collect.Lists;

/**
 * Samples from clusters that have varying frequencies but constant radius.
 */
public class LumpyData implements Sampler<Vector> {

	// size of the clusters
	private final double radius;

	// figures out which cluster to look at
	private final Sampler<Integer> cluster;

	// remembers centroids of clusters
	private final List<Sampler<Vector>> centroids = Lists.newArrayList();

	// how the centroids are generated
	private final MultiNormal centers;

	/**
	 * Samples from a lumpy distribution that acts a bit more like real data than just sampling from a normal distribution.
	 * @param dimension   The dimension of the vectors to return.
	 * @param radius      The size of the clusters we sample from.
	 * @param alpha       Controls the growth of the number of clusters.  The number of clusters will be about alpha * log(samples)
	 */
	public LumpyData(int dimension, double radius, double alpha) {
		this.centers = new MultiNormal(dimension);
		this.radius = radius;
		cluster = new ChineseRestaurant(alpha);
	}

	@Override
	public Vector sample() {
		int id = cluster.sample();
		if (id >= centroids.size()) {
			// need to invent a new cluster
			centroids.add(new MultiNormal(radius, centers.sample()));
		}
		return centroids.get(id).sample();
	}
}
