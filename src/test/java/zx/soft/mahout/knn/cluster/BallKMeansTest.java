package zx.soft.mahout.knn.cluster;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.ConstantVector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.apache.mahout.math.random.MultiNormal;
import org.junit.Test;

import zx.soft.mahout.knn.cluster.BallKmeans;
import zx.soft.mahout.knn.search.BruteSearch;

import com.google.common.collect.Lists;

public class BallKMeansTest {

	private static final int K1 = 100;

	@Test
	public void testBasicClustering() {
		List<? extends WeightedVector> data = cubishTestData(1);

		BallKmeans r = new BallKmeans(new BruteSearch(new EuclideanDistanceMeasure()), 6, 20);
		r.cluster(data);
		for (Centroid centroid : r) {
			for (int i = 0; i < 10; i++) {
				System.out.printf("%10.4f", centroid.get(i));
			}
			System.out.printf("\n");
		}
	}

	@Test
	public void testInitialization() {
		// start with super clusterable data
		List<? extends WeightedVector> data = cubishTestData(0.01);

		// just do initialization of ball k-means.  This should drop a point into each of the clusters
		BallKmeans r = new BallKmeans(new BruteSearch(new EuclideanDistanceMeasure()), 6, 20);
		r.cluster(data);

		// put the centroids into a matrix
		Matrix x = new DenseMatrix(6, 5);
		int row = 0;
		for (Centroid c : r) {
			x.viewRow(row).assign(c.viewPart(0, 5));
			row++;
		}

		// verify that each column looks right.  Should contain zeros except for a single 6.
		final Vector columnNorms = x.aggregateColumns(new VectorFunction() {
			@Override
			public double apply(Vector f) {
				// return the sum of three discrepancy measures
				return Math.abs(f.minValue()) + Math.abs(f.maxValue() - 6) + Math.abs(f.norm(1) - 6);
			}
		});
		// verify all errors are nearly zero
		assertEquals(0, columnNorms.norm(1) / columnNorms.size(), 0.1);

		// verify that the centroids are a permutation of the original ones
		SingularValueDecomposition svd = new SingularValueDecomposition(x);
		Vector s = svd.getS().viewDiagonal().assign(Functions.div(6));
		assertEquals(5, s.getLengthSquared(), 0.05);
		assertEquals(5, s.norm(1), 0.05);
	}

	private List<? extends WeightedVector> cubishTestData(double radius) {
		List<WeightedVector> data = Lists.newArrayListWithCapacity(K1 + 5000);
		int row = 0;

		MultiNormal g = new MultiNormal(radius, new ConstantVector(0, 10));
		for (int i = 0; i < K1; i++) {
			data.add(new WeightedVector(g.sample(), 1, row++));
		}

		for (int i = 0; i < 5; i++) {
			Vector m = new DenseVector(10);
			m.set(i, i == 0 ? 6 : 6);
			MultiNormal gx = new MultiNormal(radius, m);
			for (int j = 0; j < 1000; j++) {
				data.add(new WeightedVector(gx.sample(), 1, row++));
			}
		}
		return data;
	}
}
