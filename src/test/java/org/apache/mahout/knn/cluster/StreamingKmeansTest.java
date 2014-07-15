package org.apache.mahout.knn.cluster;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
// import org.apache.mahout.knn.search.Brute;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import zx.soft.mahout.knn.cluster.DataUtils;
import zx.soft.mahout.knn.cluster.StreamingKMeans;
import zx.soft.mahout.knn.search.BruteSearch;
import zx.soft.mahout.knn.search.FastProjectionSearch;
import zx.soft.mahout.knn.search.LocalitySensitiveHashSearch;
import zx.soft.mahout.knn.search.ProjectionSearch;
import zx.soft.mahout.knn.search.Searcher;
import zx.soft.mahout.knn.search.UpdatableSearcher;

@RunWith(value = Parameterized.class)
public class StreamingKmeansTest {

	private static final int NUM_DATA_POINTS = 100000;
	private static final int NUM_DIMENSIONS = 3;
	private static final int NUM_PROJECTIONS = 4;
	private static final int SEARCH_SIZE = 10;

	private static Pair<List<Centroid>, List<Centroid>> syntheticData = DataUtils.sampleMultiNormalHypercube(
			NUM_DIMENSIONS, NUM_DATA_POINTS);

	private final UpdatableSearcher searcher;
	private final boolean allAtOnce;

	public StreamingKmeansTest(UpdatableSearcher searcher, boolean allAtOnce) {
		this.searcher = searcher;
		this.allAtOnce = allAtOnce;
	}

	@Parameters
	public static List<Object[]> generateData() {
		return Arrays.asList(new Object[][] {
				{ new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), true },
				{ new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), true },
				{ new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), SEARCH_SIZE), true },
				{ new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), false },
				{ new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), false },
				{ new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), SEARCH_SIZE), false } });
	}

	@Test
	public void testClustering() {
		StreamingKMeans clusterer = new StreamingKMeans(searcher, 1 << NUM_DIMENSIONS,
				DataUtils.estimateDistanceCutoff(syntheticData.getFirst()));
		long startTime = System.currentTimeMillis();
		if (allAtOnce) {
			clusterer.cluster(syntheticData.getFirst());
		} else {
			for (Centroid datapoint : syntheticData.getFirst()) {
				clusterer.cluster(datapoint);
			}
		}
		long endTime = System.currentTimeMillis();

		System.out.printf("Total number of clusters %d\n", clusterer.getCentroids().size());

		assertEquals("Total weight not preserved", totalWeight(syntheticData.getFirst()),
				totalWeight(clusterer.getCentroids()), 1e-9);

		// and verify that each corner of the cube has a centroid very nearby
		for (Vector mean : syntheticData.getSecond()) {
			WeightedThing<Vector> v = searcher.search(mean, 1).get(0);
			assertTrue(v.getWeight() < 0.05);
		}
		double clusterTime = (endTime - startTime) / 1000.0;
		System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n", searcher.getClass().getName(), clusterTime,
				clusterTime / syntheticData.getFirst().size() * 1e6);

		// verify that the total weight of the centroids near each corner is correct
		double[] cornerWeights = new double[1 << NUM_DIMENSIONS];
		Searcher trueFinder = new BruteSearch(new EuclideanDistanceMeasure());
		for (Vector trueCluster : syntheticData.getSecond()) {
			trueFinder.add(trueCluster);
		}
		for (Centroid centroid : clusterer.getCentroidsIterable()) {
			WeightedThing<Vector> closest = trueFinder.search(centroid, 1).get(0);
			cornerWeights[((Centroid) closest.getValue()).getIndex()] += centroid.getWeight();
		}
		int expectedNumPoints = NUM_DATA_POINTS / (1 << NUM_DIMENSIONS);
		for (double v : cornerWeights) {
			System.out.printf("%f ", v);
		}
		System.out.println();
		for (double v : cornerWeights) {
			assertEquals(expectedNumPoints, v, 0);
		}
	}

	private double totalWeight(Iterable<? extends Vector> data) {
		double sum = 0;
		for (Vector row : data) {
			if (row instanceof WeightedVector) {
				sum += ((WeightedVector) row).getWeight();
			} else {
				sum++;
			}
		}
		return sum;
	}

}
