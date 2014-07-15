package org.apache.mahout.knn.search;

import java.util.BitSet;
import java.util.List;

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.Normal;
import org.apache.mahout.math.random.WeightedThing;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.junit.Assert;
import org.junit.Test;

import zx.soft.mahout.knn.search.BruteSearch;
import zx.soft.mahout.knn.search.HashedVector;
import zx.soft.mahout.knn.search.LocalitySensitiveHashSearch;

public class LocalitySensitiveHashSearchTest {

	@Test
	public void testNormal() {
		Matrix testData = new DenseMatrix(100000, 10);
		final Normal gen = new Normal();
		testData.assign(gen);

		final EuclideanDistanceMeasure distance = new EuclideanDistanceMeasure();
		BruteSearch ref = new BruteSearch(distance);
		ref.addAllMatrixSlicesAsWeightedVectors(testData);

		LocalitySensitiveHashSearch cut = new LocalitySensitiveHashSearch(distance, 10);
		cut.addAllMatrixSlicesAsWeightedVectors(testData);

		cut.setSearchSize(200);
		cut.resetEvaluationCount();

		System.out.printf("speedup,q1,q2,q3\n");

		for (int i = 0; i < 12; i++) {
			double strategy = (i - 1.0) / 10.0;
			cut.setRaiseHashLimitStrategy(strategy);
			OnlineSummarizer t1 = evaluateStrategy(testData, ref, cut);
			int evals = cut.resetEvaluationCount();
			final double speedup = 10e6 / evals;
			System.out
					.printf("%.1f,%.2f,%.2f,%.2f\n", speedup, t1.getQuartile(1), t1.getQuartile(2), t1.getQuartile(3));
			Assert.assertTrue(t1.getQuartile(2) > 0.45);
			Assert.assertTrue(speedup > 4 || t1.getQuartile(2) > 0.9);
			Assert.assertTrue(speedup > 15 || t1.getQuartile(2) > 0.8);
		}
	}

	private OnlineSummarizer evaluateStrategy(Matrix testData, BruteSearch ref, LocalitySensitiveHashSearch cut) {
		OnlineSummarizer t1 = new OnlineSummarizer();

		for (int i = 0; i < 100; i++) {
			final Vector q = testData.viewRow(i);
			List<WeightedThing<Vector>> v1 = cut.search(q, 150);
			BitSet b1 = new BitSet();
			for (WeightedThing<Vector> v : v1) {
				b1.set(((WeightedVector) v.getValue()).getIndex());
			}

			List<WeightedThing<Vector>> v2 = ref.search(q, 100);
			BitSet b2 = new BitSet();
			for (WeightedThing<Vector> v : v2) {
				b2.set(((WeightedVector) v.getValue()).getIndex());
			}

			b1.and(b2);
			t1.add(b1.cardinality());
		}
		return t1;
	}

	@Test
	public void testDotCorrelation() {
		final Normal gen = new Normal();

		Matrix projection = new DenseMatrix(64, 10);
		projection.assign(gen);

		Vector query = new DenseVector(10);
		query.assign(gen);
		long qhash = HashedVector.computeHash64(query, projection);

		int count[] = new int[65];
		Vector v = new DenseVector(10);
		for (int i = 0; i < 500000; i++) {
			v.assign(gen);
			long hash = HashedVector.computeHash64(v, projection);
			final int bitDot = Long.bitCount(qhash ^ hash);
			count[bitDot]++;
			if (count[bitDot] < 200) {
				System.out.printf("%d, %.3f\n", bitDot,
						v.dot(query) / Math.sqrt(v.getLengthSquared() * query.getLengthSquared()));
			}
		}
	}
}
