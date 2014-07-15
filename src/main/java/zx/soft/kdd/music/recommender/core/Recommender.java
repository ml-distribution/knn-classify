package zx.soft.kdd.music.recommender.core;

import java.io.FileNotFoundException;

/**
 * 推荐接口
 * 
 * @author wanggang
 *
 */
public interface Recommender {

	public void createNeighborhoods();

	public void recommendSong(String activeUserFile, double threshold) throws FileNotFoundException;

}
