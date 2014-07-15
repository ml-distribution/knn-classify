package zx.soft.knn.classify.driver;

import org.apache.hadoop.util.ProgramDriver;

import zx.soft.kdd.music.recommender.KDDMusicRecommender;

public class KnnClassifyDriver {

	public static void main(String argv[]) {

		int exitCode = -1;
		ProgramDriver pgd = new ProgramDriver();
		try {
			pgd.addClass("kDDMusicRecommender", KDDMusicRecommender.class, "基于KNN的KDD音乐推荐算法示例");
			pgd.driver(argv);
			// Success
			exitCode = 0;
		} catch (Throwable e) {
			e.printStackTrace();
		}

		System.exit(exitCode);
	}

}
