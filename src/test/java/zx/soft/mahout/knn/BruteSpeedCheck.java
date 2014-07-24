package zx.soft.mahout.knn;

import java.util.List;

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.ConstantVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.Sampler;

import zx.soft.mahout.knn.search.BruteSearch;

import com.google.common.collect.Lists;

public class BruteSpeedCheck {

	private static final int VECTOR_DIMENSION = 250;
	private static final int REFERENCE_SIZE = 10000;
	private static final int QUERY_SIZE = 100;

	public static void main(String[] args) {
		Sampler<Vector> rand = new MultiNormal(new ConstantVector(1, VECTOR_DIMENSION));
		List<WeightedVector> referenceVectors = Lists.newArrayListWithExpectedSize(REFERENCE_SIZE);
		for (int i = 0; i < REFERENCE_SIZE; ++i) {
			referenceVectors.add(new WeightedVector(rand.sample(), 1, i));
		}
		System.out.printf("Generated reference matrix.\n");

		List<WeightedVector> queryVectors = Lists.newArrayListWithExpectedSize(QUERY_SIZE);
		for (int i = 0; i < QUERY_SIZE; ++i) {
			queryVectors.add(new WeightedVector(rand.sample(), 1, i));
		}
		System.out.printf("Generated query matrix.\n");

		for (int threads : new int[] { 1, 2, 3, 4, 5, 6, 10, 20, 50 }) {
			for (int block : new int[] { 1, 10, 50 }) {
				BruteSearch search = new BruteSearch(new EuclideanDistanceMeasure());
				search.addAll(referenceVectors);
				long t0 = System.nanoTime();
				search.search(queryVectors, block, threads);
				long t1 = System.nanoTime();
				System.out.printf("%d\t%d\t%.2f\n", threads, block, (t1 - t0) / 1e9);
			}
		}
	}

}
