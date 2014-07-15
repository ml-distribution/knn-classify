package zx.soft.kdd.music.recommender.db;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import zx.soft.kdd.music.recommender.KDDMusicRecommender;

/**
 * @author sarahejones, sns
 */
public class Similarities implements Iterable<Similarity> {

	private final int k;
	private final ArrayList<Similarity> neighbors;

	public Similarities() {
		k = KDDMusicRecommender.getOptions().getK();
		neighbors = new ArrayList<Similarity>();
	}

	public int getK() {
		return k;
	}

	public void insert(Similarity is) {
		int index = Collections.binarySearch(neighbors, is);
		if (index < 0) {
			neighbors.add(-index - 1, is);
		}
		while (neighbors.size() > k) {
			neighbors.remove(0);
		}
	}

	public void print() {
		for (Similarity is : neighbors) {
			System.out.println("-" + "\t" + is.getNeighborSong().getID() + "\t" + is.getSimilarity());
		}
	}

	public double getSimilarity(Song song) {
		for (Similarity sim : neighbors) {
			if (song.getID() == sim.getNeighborSong().getID()) {
				return sim.getSimilarity();
			}
		}
		return 0;
	}

	public boolean contains(Song song) {
		for (Similarity sim : neighbors) {
			if (song.getID() == sim.getNeighborSong().getID()) {
				return true;
			}
		}
		return false;
	}

	@Override
	public Iterator<Similarity> iterator() {
		return neighbors.iterator();
	}
}
