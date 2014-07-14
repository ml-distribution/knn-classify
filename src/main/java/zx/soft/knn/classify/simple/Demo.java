package zx.soft.knn.classify.simple;

public class Demo {

	public static void main(String[] args) {

		String str = "0 938.935 0	-39.7896 0 298.622 0	0	0";
		String[] strs = str.split("\\s");
		for (String s : strs) {
			System.out.println(s);
		}
	}
}
