package zx.soft.mahout.knn.search;

import java.util.List;

import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;

import com.google.common.collect.Lists;

/**
 * Describes how to search a bunch of vectors.
 * The vectors can be of any type (weighted, sparse, ...) but only the values of the vector  matter
 * when searching (weights, indices, ...) will not.
 *
 * When iterating through a Searcher, the Vectors added to it are returned.
 */
public abstract class Searcher implements Iterable<Vector> {

	protected DistanceMeasure distanceMeasure;

	public Searcher(DistanceMeasure distanceMeasure) {
		this.distanceMeasure = distanceMeasure;
	}

	public DistanceMeasure getDistanceMeasure() {
		return distanceMeasure;
	}

	/**
	 * Add a new Vector to the Searcher that will be checked when getting
	 * the nearest neighbors.
	 *
	 * The vector IS NOT CLONED. Do not modify the vector externally otherwise the internal
	 * Searcher data structures could be invalidated.
	 */
	public abstract void add(Vector v);

	/**
	 * Returns the number of WeightedVectors being searched for nearest neighbors.
	 */
	public abstract int size();

	/**
	 * When querying the Searcher for the closest vectors, a list of WeightedThing<Vector>s is
	 * returned. The value of the WeightedThing is the neighbor and the weight is the
	 * the distance (calculated by some metric - see a concrete implementation) between the query
	 * and neighbor.
	 * The actual type of vector in the pair is the same as the vector added to the Searcher.
	 */
	public abstract List<WeightedThing<Vector>> search(Vector query, int limit);

	public List<List<WeightedThing<Vector>>> search(Iterable<? extends Vector> queries, int limit) {
		List<List<WeightedThing<Vector>>> results = Lists.newArrayList();
		for (Vector query : queries)
			results.add(search(query, limit));
		return results;
	}

	/**
	 * Adds all the data elements in the Searcher.
	 *
	 * @param data an iterable of WeightedVectors to add.
	 */
	public void addAll(Iterable<? extends Vector> data) {
		for (Vector v : data) {
			add(v);
		}
	}

	/**
	 * Adds all the data elements in the Searcher.
	 *
	 * @param data an iterable of MatrixSlices to add.
	 */
	public void addAllMatrixSlices(Iterable<MatrixSlice> data) {
		for (MatrixSlice slice : data) {
			add(slice.vector());
		}
	}

	public void addAllMatrixSlicesAsWeightedVectors(Iterable<MatrixSlice> data) {
		for (MatrixSlice slice : data) {
			add(new WeightedVector(slice.vector(), 1, slice.index()));
		}
	}

	public boolean remove(Vector v, double epsilon) {
		throw new UnsupportedOperationException("Can't remove a vector from a " + this.getClass().getName());
	}

	public void clear() {
		throw new UnsupportedOperationException("Can't remove vectors from a " + this.getClass().getName());
	}

}
