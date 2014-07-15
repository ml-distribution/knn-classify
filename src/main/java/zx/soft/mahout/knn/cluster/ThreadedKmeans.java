package zx.soft.mahout.knn.cluster;

public class ThreadedKmeans {
	/*
	    public Searcher cluster(final DistanceMeasure distance, List<Iterable<MatrixSlice>> data, final int maxClusters, final int threads, final OldStreamingKmeans.SearchFactory centroidFactory) throws InterruptedException, ExecutionException {
	        // initialize scale
	        int i = 0;
	        final int width = data.get(0).iterator().next().vector().size();
	        Matrix sample = new DenseMatrix(100, width);
	        for (Iterable<MatrixSlice> split : data) {
	            sample = sampleRows(sample, i, Iterables.limit(split, 1000), 100);
	            i += 100;
	        }

	        List<Callable<Searcher>> tasks = Lists.newArrayList();
	        ExecutorService pool = Executors.newFixedThreadPool(threads);
	        for (final Iterable<MatrixSlice> split : data) {
	            tasks.add(new Callable<Searcher>() {
	                @Override
	                public Searcher call() {
	                    return new OldStreamingKmeans().cluster(split, maxClusters, centroidFactory);
	                }
	            });
	        }
	        List<Future<Searcher>> results = pool.invokeAll(tasks);
	        pool.shutdown();

	        List<MatrixSlice> raw = Lists.newArrayList();
	        for (Future<Searcher> result : results) {
	            Iterables.addAll(raw, result.get());
	        }

	        return new OldStreamingKmeans().cluster(raw, data.size() * maxClusters, centroidFactory);
	    }

	    public static List<Iterable<MatrixSlice>> split(Iterable<MatrixSlice> data, int threads) {
	        List<Iterable<MatrixSlice>> r = Lists.newArrayList();
	        int size = Iterables.size(data);
	        int block = (size + threads - 1) / threads;

	        for (int start = 0; start < size; start += block) {
	            final Iterable<MatrixSlice> split = Iterables.limit(Iterables.skip(data, start), (Math.min(start + block, size) - start));
	            r.add(split);
	        }
	        return r;
	    }

	    private Matrix sampleRows(Matrix r, int soFar, Iterable<MatrixSlice> data, int n) {
	        Random rand = new Random();
	        int i = soFar;
	        for (MatrixSlice row : data) {
	            if (i < n) {
	                r.viewRow(i).assign(row.vector());
	            } else {
	                int k = rand.nextInt(n);
	                if (k < n) {
	                    r.viewRow(k).assign(row.vector());
	                }
	            }
	            i++;
	        }

	        return r;
	    }
	    */
}
