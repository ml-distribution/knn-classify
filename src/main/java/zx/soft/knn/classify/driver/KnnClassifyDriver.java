package zx.soft.knn.classify.driver;

import org.apache.hadoop.util.ProgramDriver;

public class KnnClassifyDriver {

	public static void main(String argv[]) {

		int exitCode = -1;
		ProgramDriver pgd = new ProgramDriver();
		try {
			//			pgd.addClass("kMeansClusterDistribute", KMeansClusterDistribute.class, "分布式KMeans聚类算法");
			//			pgd.addClass("kMeansCore", KMeansCore.class, "简单KMeans聚类算法");
			pgd.driver(argv);
			// Success
			exitCode = 0;
		} catch (Throwable e) {
			e.printStackTrace();
		}

		System.exit(exitCode);
	}

}
