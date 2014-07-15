package zx.soft.mahout.knn.search;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;

/**
 * Decorates a weighted vector with a locality sensitive hash.
 */
public class HashedVector extends WeightedVector {
	protected static int INVALID_INDEX = -1;
	private final long hash;

	public HashedVector(Vector v, long hash, int index) {
		super(v, 1, index);
		this.hash = hash;
	}

	public HashedVector(Vector v, Matrix projection, int index, long mask) {
		super(v, 1, index);
		this.hash = mask & computeHash64(v, projection);
	}

	public HashedVector(WeightedVector v, Matrix projection, long mask) {
		super(v.getVector(), v.getWeight(), v.getIndex());
		this.hash = mask;
	}

	public static int computeHash(Vector v, Matrix projection) {
		int hash = 0;
		for (Element element : projection.times(v).all()) {
			if (element.get() > 0) {
				hash += 1 << element.index();
			}
		}
		return hash;
	}

	public static long computeHash64(Vector v, Matrix projection) {
		long hash = 0;
		for (Element element : projection.times(v).all()) {
			if (element.get() > 0) {
				hash += 1L << element.index();
			}
		}
		return hash;
	}

	public static HashedVector hash(WeightedVector v, Matrix projection) {
		return hash(v, projection, 0);
	}

	public static HashedVector hash(WeightedVector v, Matrix projection, long mask) {
		return new HashedVector(v, projection, mask);
	}

	public int xor(HashedVector v) {
		return Long.bitCount(v.getHash() ^ hash);
	}

	public long getHash() {
		return hash;
	}

	@Override
	public String toString() {
		return String.format("index=%d, hash=%08x, v=%s", getIndex(), hash, getVector());
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof HashedVector)) {
			return o instanceof Vector && this.minus((Vector) o).norm(1) == 0;
		} else {
			HashedVector v = (HashedVector) o;
			return v.hash == this.hash && this.minus(v).norm(1) == 0;
		}
	}

	@Override
	public int hashCode() {
		int result = super.hashCode();
		result = 31 * result + (int) (hash ^ (hash >>> 32));
		return result;
	}
}
