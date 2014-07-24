package zx.soft.mahout.knn.search;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Before;
import org.junit.Test;

import zx.soft.mahout.knn.search.BruteSearch;
import zx.soft.mahout.knn.search.UpdatableSearcher;

import com.google.common.collect.Lists;

public class BruteSearchTest extends AbstractSearchTest {

	private static Iterable<MatrixSlice> data;

	@Before
	public void fillData() {
		data = randomData();
	}

	@Override
	public Iterable<MatrixSlice> testData() {
		return data;
	}

	@Override
	public UpdatableSearcher getSearch(int n) {
		return new BruteSearch(new EuclideanDistanceMeasure());
	}

	@Test
	public void testMatrixSearch() {
		List<WeightedVector> referenceVectors = Lists.newArrayListWithExpectedSize(8);
		BruteSearch searcher = new BruteSearch(new EuclideanDistanceMeasure());
		for (int i = 0; i < 8; i++) {
			referenceVectors.add(new WeightedVector(new DenseVector(new double[] { 0.125 * (i & 4), i & 2, i & 1 }), 1,
					i));
			searcher.add(referenceVectors.get(referenceVectors.size() - 1));
		}

		final List<List<WeightedThing<Vector>>> searchResults = searcher.search(referenceVectors, 3);
		for (List<WeightedThing<Vector>> r : searchResults) {
			assertEquals(0, r.get(0).getWeight(), 1e-8);
			assertEquals(0.5, r.get(1).getWeight(), 1e-8);
			assertEquals(1, r.get(2).getWeight(), 1e-8);
		}
	}

}
