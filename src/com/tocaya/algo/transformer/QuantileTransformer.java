package com.tocaya.algo.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.log4j.Logger;

/**
 * A Quantile Transformer adapted from scikit-learn's <a href=
 * "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html">sklearn.preprocessing.QuantileTransformer</a>.
 * <br/>
 * <br/>
 * <b>NOTE:</b> This implementation only supports <code>subsample</code> == <code>nSamples</code>
 * and will throw an <code>UnsupportedOperationException</code> otherwise. <br/>
 * <br/>
 * Additionally, this implementation only supports a single input feature/column.
 *
 */
public class QuantileTransformer implements ITransformer, Serializable, Comparable<QuantileTransformer> {

	private static final Logger log = Logger.getLogger(QuantileTransformer.class);

	private static final long serialVersionUID = 1537967572554342221L;

	private static double LOWER_BOUND_Y = 0;
	private static double UPPER_BOUND_Y = 1;
	private static int LIKELY_IN_CACHE_SIZE = 8;

	private int nQuantiles;
	private int subsample;
	private double[] references;
	private double[] quantiles;
	private double lowerBoundX;
	private double upperBoundX;

	private static Set<QuantileTransformer> transformerCache = new HashSet<>();

	/**
	 * Stores transformers in a private cache. When an existing transformer is found that results in
	 * <code>compareTo==0</code> with <code>transformer</code> then the cached object will be
	 * returned. Otherwise <code>transformer</code> will be returned. <br/>
	 * <br/>
	 * Instead of making the <code>QuantileTransformer</code> constructor private, set your local
	 * transformer to the result of this cache method to prevent duplicating transformers in memory.
	 * <br/>
	 * <br/>
	 * The implementation synchronizes on the transformer cache so this method <b>is</b>
	 * thread-safe.
	 * 
	 * @param transformer
	 *            The transformer
	 * @return A cached copy of the transformer
	 */
	public static QuantileTransformer getQuantileTransformerFromCache(QuantileTransformer transformer) {
		if (transformer == null) {
			return null;
		}
		synchronized (transformerCache) {
			for (QuantileTransformer qt : transformerCache) {
				if (qt.compareTo(transformer) == 0) {
					return qt;
				}
			}
			transformerCache.add(transformer);
			return transformer;
		}
	}

	/**
	 * Clears the cache of transformers. <br/>
	 * <br/>
	 * The implementation synchronizes on the transformer cache so this method <b>is</b>
	 * thread-safe.
	 */
	public static void clearCache() {
		synchronized (transformerCache) {
			log.info("Clearing quantile transformer cache");
			transformerCache.clear();
		}
	}

	/**
	 * Create an empty quantile transformer with the specified number of quantiles and subsample.
	 * 
	 * @param nQuantiles
	 *            Number of quantiles
	 * @param subsample
	 *            Number of subsamples
	 */
	public QuantileTransformer(int nQuantiles, int subsample) {
		this.nQuantiles = nQuantiles;
		this.subsample = subsample;
	}

	/**
	 * Construct a quantile transformer from an existing internal representation. <br/>
	 * <br/>
	 * Only not use this constructor if you have the internal state of an existing quantile
	 * transformer! <br/>
	 * <br/>
	 * This constructor is useful from porting a scikit-learn <a href=
	 * "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html">sklearn.preprocessing.QuantileTransformer</a>
	 * since the implementations share the same internal objects.
	 * 
	 * @param nQuantiles
	 *            Number of quantiles
	 * @param subsample
	 *            Number of subsamples
	 * @param references
	 *            The references
	 * @param quantiles
	 *            The quantiles
	 */
	public QuantileTransformer(int nQuantiles, int subsample, double[] references, double[] quantiles) {
		this.nQuantiles = nQuantiles;
		this.subsample = subsample;
		setReferences(references);
		setQuantiles(quantiles);
	}

	@Override
	public <I> void fit(I x) {
		if (double[].class.equals(x.getClass())) {
			fit((double[]) x);
		} else if (double[].class.equals(x.getClass())) {
			fit(flatten((double[][]) x));
		} else {
			throw new UnsupportedOperationException("The input type " + x.getClass().getName() + " is not supported for this operation");
		}
	}

	@Override
	public <I, O> O transform(I x, Class<O> returnType, boolean rowMajorOrder) {
		if (Double.class.equals(x.getClass())) {
			return transform((double) x, returnType);
		} else if (Float.class.equals(x.getClass())) {
			return transform(((Float) x).doubleValue(), returnType);
		} else if (double[].class.equals(x.getClass())) {
			return transform((double[]) x, returnType);
		} else if (float[].class.equals(x.getClass())) {
			log.info("Performing cast before transform, this is not optimal");
			double[] input = toDoubleArray((float[]) x);
			return transform(input, returnType);
		} else {
			throw new UnsupportedOperationException("The input type " + x.getClass().getName() + " is not supported for this operation");
		}
	}

	private void setQuantiles(double[] quantiles) {
		this.quantiles = quantiles;
		this.lowerBoundX = this.quantiles[0];
		this.upperBoundX = this.quantiles[this.quantiles.length - 1];
	}

	private void setReferences(double[] references) {
		this.references = references;
	}

	@SuppressWarnings("unchecked")
	private <T> T transform(double x, Class<T> returnType) {
		Double result;
		if (x == lowerBoundX) {
			result = LOWER_BOUND_Y;
		} else if (x == upperBoundX) {
			result = UPPER_BOUND_Y;
		} else if (!Double.isNaN(x)) {
			result = 0.5 * (interp(x, this.quantiles, this.references) - interpReverseNegated(x, this.quantiles, this.references));
		} else {
			result = x;
		}
		if (Double.class.equals(returnType) || double.class.equals(returnType)) {
			return (T) result;
		} else if (Float.class.equals(returnType) || float.class.equals(returnType)) {
			return (T) new Float(result.floatValue());
		} else {
			throw new UnsupportedOperationException("The returnType " + returnType.getName() + " is not supported for this operation");
		}
	}

	@SuppressWarnings("unchecked")
	private <T> T transform(double[] x, Class<T> returnType) {
		final int xLen = x.length;
		double[] result = new double[xLen];
		for (int i = 0; i < xLen; i++) {
			if (Double.isNaN(x[i])) {
				result[i] = Double.NaN;
			} else {
				result[i] = 0.5
						* (interp(x[i], this.quantiles, this.references) - interpReverseNegated(x[i], this.quantiles, this.references));
			}
		}
		eqApply(x, lowerBoundX, result, LOWER_BOUND_Y);
		eqApply(x, upperBoundX, result, UPPER_BOUND_Y);
		if (double[].class.equals(returnType)) {
			return (T) result;
		} else if (float[].class.equals(returnType)) {
			return (T) toFloatArray(result);
		} else {
			throw new UnsupportedOperationException("The returnType " + returnType.getName() + " is not supported for this operation");
		}
	}

	@SuppressWarnings("unused")
	private double[] interp(double[] x, double[] xp, double[] fp) {
		double[] afp = fp;
		double[] axp = xp;
		double[] ax = x;
		int lenxp = xp.length;
		double[] af = new double[ax.length];
		int lenx = ax.length;
		double[] dy = afp;
		double[] dx = axp;
		double[] dz = ax;
		double[] dres = af;
		double lval = fp[0];
		double rval = fp[lenxp - 1];
		if (lenxp == 1) {
			double xp_val = dx[0];
			double fp_val = dy[0];
			for (int i = 0; i < lenx; ++i) {
				double x_val = dz[i];
				dres[i] = (x_val < xp_val) ? lval : ((x_val > xp_val) ? rval : fp_val);
			}
		} else {
			int j = 0;
			double[] slopes = null;
			/* only pre-calculate slopes if there are relatively few of them. */
			if (lenxp <= lenx) {
				slopes = new double[lenxp - 1];
			}
			if (slopes != null) {
				for (int i = 0; i < lenxp - 1; ++i) {
					slopes[i] = (dy[i + 1] - dy[i]) / (dx[i + 1] - dx[i]);
				}
			}
			for (int i = 0; i < lenx; ++i) {
				double x_val = dz[i];

				if (Double.isNaN(x_val)) {
					dres[i] = x_val;
					continue;
				}

				j = binarySearchWithGuess(x_val, dx, lenxp, j);
				if (j == -1) {
					dres[i] = lval;
				} else if (j == lenxp) {
					dres[i] = rval;
				} else if (j == lenxp - 1) {
					dres[i] = dy[j];
				} else if (dx[j] == x_val) {
					/* Avoid potential non-finite interpolation */
					dres[i] = dy[j];
				} else {
					double slope = (slopes != null) ? slopes[j] : (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j]);

					/* If we get nan in one direction, try the other */
					dres[i] = slope * (x_val - dx[j]) + dy[j];
					if ((Double.isNaN(dres[i]))) {
						dres[i] = slope * (x_val - dx[j + 1]) + dy[j + 1];
						if ((Double.isNaN(dres[i])) && dy[j] == dy[j + 1]) {
							dres[i] = dy[j];
						}
					}
				}
			}
		}
		return af;
	}

	private double interp(double x, double[] xp, double[] fp) {
		final int lenxp = xp.length;
		double dres;
		if (lenxp == 1) {
			double xp_val = xp[0];
			double fp_val = fp[0];
			dres = (x < xp_val) ? fp[0] : ((x > xp_val) ? fp[lenxp - 1] : fp_val);
		} else {
			int j = 0;
			double[] slopes = null;
			// only pre-calculate slopes if there are relatively few of them
			if (lenxp <= 1) {
				slopes = new double[lenxp - 1];
			}
			if (slopes != null) {
				for (int i = 0; i < lenxp - 1; ++i) {
					slopes[i] = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]);
				}
			}
			j = binarySearchWithGuess(x, xp, lenxp, j);
			if (j == -1) {
				dres = fp[0];
			} else if (j == lenxp) {
				dres = fp[lenxp - 1];
			} else if (j == lenxp - 1) {
				dres = fp[j];
			} else if (xp[j] == x) {
				// Avoid potential non-finite interpolation
				dres = fp[j];
			} else {
				double slope = (slopes != null) ? slopes[j] : (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
				// If we get nan in one direction, try the other
				dres = slope * (x - xp[j]) + fp[j];
				if (Double.isNaN(dres)) {
					dres = slope * (x - xp[j + 1]) + fp[j + 1];
					if (Double.isNaN(dres) && fp[j] == fp[j + 1]) {
						dres = fp[j];
					}
				}
			}
		}
		return dres;
	}

	private double interpReverseNegated(double x, double[] xp, double[] fp) {
		x = -x;
		final int lenxp = xp.length;
		final int lenxpminusone = lenxp - 1;
		final int lenfpminusone = fp.length - 1;
		double dres;
		if (lenxp == 1) {
			double xp_val = -xp[lenxpminusone];
			double fp_val = -fp[lenfpminusone];
			dres = (x < xp_val) ? -fp[lenfpminusone] : ((x > xp_val) ? -fp[lenfpminusone - (lenxp - 1)] : fp_val);
		} else {
			int j = 0;
			double[] slopes = null;
			// only pre-calculate slopes if there are relatively few of them
			if (lenxp <= 1) {
				slopes = new double[lenxp - 1];
			}
			if (slopes != null) {
				for (int i = 0; i < lenxp - 1; ++i) {
					slopes[i] = (-fp[lenfpminusone - (i + 1)] - -fp[lenfpminusone - i])
							/ (-xp[lenxpminusone - (i + 1)] - -xp[lenxpminusone - i]);
				}
			}
			j = binarySearchWithGuessReverseNegated(x, xp, lenxp, j, lenxpminusone);
			if (j == -1) {
				dres = -fp[lenfpminusone];
			} else if (j == lenxp) {
				dres = -fp[lenfpminusone - (lenxp - 1)];
			} else if (j == lenxp - 1) {
				dres = -fp[lenfpminusone - j];
			} else if (-xp[lenxpminusone - j] == x) {
				// Avoid potential non-finite interpolation
				dres = -fp[lenfpminusone - j];
			} else {
				double slope = (slopes != null) ? slopes[j]
						: (-fp[lenfpminusone - (j + 1)] - -fp[lenfpminusone - j]) / (-xp[lenxpminusone - (j + 1)] - -xp[lenxpminusone - j]);
				// If we get nan in one direction, try the other
				dres = slope * (x - -xp[lenxpminusone - j]) + -fp[lenfpminusone - j];
				if (Double.isNaN(dres)) {
					dres = slope * (x - -xp[lenxpminusone - (j + 1)]) + -fp[lenfpminusone - (j + 1)];
					if (Double.isNaN(dres) && -fp[lenfpminusone - j] == -fp[lenfpminusone - (j + 1)]) {
						dres = -fp[lenfpminusone - j];
					}
				}
			}
		}
		return dres;
	}

	private void fit(double[] x) {
		int nSamples = x.length;
		this.nQuantiles = Math.max(1, Math.min(this.nQuantiles, nSamples));
		setReferences(linspace(0, 1, this.nQuantiles, true));
		denseFit(x);
	}

	private void denseFit(double[] x) {
		int nSamples = x.length;
		double[] references = this.references;
		double[] quantiles;
		if (this.subsample < nSamples) {
			throw new UnsupportedOperationException("See line 2272 in sklearn.preprocessing._data.py for implementation details.");
		}
		quantiles = nanPercentile(x, references);
		setQuantiles(quantiles);
	}

	private static void partition(double[] a, int[] indicesAll) {
		Arrays.sort(a);
	}

	private static double[] linspace(double start, double stop, int num, boolean endpoint) {
		int div = endpoint ? (num - 1) : num;
		double delta = stop - start;
		double[] y = new double[num];
		for (int i = 0; i < y.length; i++) {
			y[i] = i;
		}
		double step;
		if (div > 0) {
			step = delta / div;
			if (step == 0) {
				y = divide(y, div);
				y = multiply(y, delta);
			} else {
				y = multiply(y, step);
			}
		} else {
			step = Double.NaN;
			y = multiply(y, delta);
		}
		y = add(y, start);
		if (endpoint && num > 1) {
			y[y.length - 1] = stop;
		}
		return y;
	}

	private static double[] nanPercentile(double[] a, double[] q) {
		a = removeNan(a);
		int Nx = a.length;
		double[] indices = multiply(q, (Nx - 1));
		int[] indicesBelow = floor(indices);
		int[] indicesAbove = add(indicesBelow, 1);
		for (int i = 0; i < indicesAbove.length; i++) {
			if (indicesAbove[i] > Nx - 1) {
				indicesAbove[i] = Nx - 1;
			}
		}
		partition(a, null);
		double[] weightsAbove = subtract(indices, indicesBelow);
		double[] xBelow = take(a, indicesBelow);
		double[] xAbove = take(a, indicesAbove);
		double[] r = lerp(xBelow, xAbove, weightsAbove);
		return r;
	}

	private static double[] lerp(double[] a, double[] b, double[] t) {
		double[] bMinusA = subtract(b, a);
		double[] lerpInterpolation = add(a, multiply(t, bMinusA));
		for (int i = 0; i < b.length; i++) {
			if (t[i] >= 0.5) {
				lerpInterpolation[i] = b[i] - bMinusA[i] * (1 - t[i]);
			}
		}
		return lerpInterpolation;
	}

	private static int[] add(int[] x1, int x2) {
		final int len = x1.length;
		int[] out = new int[len];
		for (int i = 0; i < len; i++) {
			out[i] = x1[i] + x2;
		}
		return out;
	}

	private static double[] add(double[] x1, double[] x2) {
		final int len = x1.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = x1[i] + x2[i];
		}
		return out;
	}

	private static double[] multiply(double[] x1, double[] x2) {
		final int len = x1.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = x1[i] * x2[i];
		}
		return out;
	}

	private static double[] subtract(double[] x1, int[] x2) {
		final int len = x1.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = x1[i] - x2[i];
		}
		return out;
	}

	private static double[] subtract(double[] x1, double[] x2) {
		final int len = x1.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = x1[i] - x2[i];
		}
		return out;
	}

	private static double[] add(double[] a, double d) {
		final int len = a.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = a[i] + d;
		}
		return out;
	}

	private static double[] multiply(double[] a, double d) {
		final int len = a.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = a[i] * d;
		}
		return out;
	}

	private static double[] divide(double[] a, double d) {
		final int len = a.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = a[i] / d;
		}
		return out;
	}

	private static double[] take(double[] a, int[] indicesBelow) {
		final int len = indicesBelow.length;
		double[] out = new double[len];
		for (int i = 0; i < len; i++) {
			out[i] = a[indicesBelow[i]];
		}
		return out;
	}

	private static int[] floor(double[] indices) {
		final int len = indices.length;
		int[] floors = new int[len];
		for (int i = 0; i < len; i++) {
			floors[i] = (int) Math.floor(indices[i]);
		}
		return floors;
	}

	private static double[] removeNan(double[] a) {
		final int len = a.length;
		List<Double> noNan = new ArrayList<Double>(len);
		for (int i = 0; i < len; i++) {
			if (!Double.isNaN(a[i])) {
				noNan.add(a[i]);
			}
		}
		return toDoubleArray(noNan.toArray(), true);
	}

	private static double[] toDoubleArray(Object[] boxedArray, boolean throwNPE) {
		int len = boxedArray.length;
		double[] array = new double[len];
		for (int i = 0; i < len; i++) {
			array[i] = boxedArray[i] == null && !throwNPE ? Double.NaN : ((Number) (boxedArray[i])).doubleValue();
		}
		return array;
	}

	private static double[] toDoubleArray(float[] arr) {
		if (arr == null)
			return null;
		final int n = arr.length;
		double[] ret = new double[n];
		for (int i = 0; i < n; i++) {
			ret[i] = (double) arr[i];
		}
		return ret;
	}

	private static double[] flatten(double[][] x) {
		return Arrays.stream(x).flatMapToDouble(Arrays::stream).toArray();
	}

	private static float[] toFloatArray(double[] arr) {
		if (arr == null)
			return null;
		final int n = arr.length;
		float[] ret = new float[n];
		for (int i = 0; i < n; i++) {
			ret[i] = (float) arr[i];
		}
		return ret;
	}

	private static int binarySearchWithGuess(double key, double[] arr, int len, int guess) {
		int imin = 0;
		int imax = len;
		// Handle keys outside of the arr range first
		if (key > arr[len - 1]) {
			return len;
		} else if (key < arr[0]) {
			return -1;
		}
		// If len <= 4 use linear search. From above we know key >= arr[0] when we start.
		if (len <= 4) {
			return linearSearch(key, arr, len, 1);
		}
		if (guess > len - 3) {
			guess = len - 3;
		}
		if (guess < 1) {
			guess = 1;
		}
		// check most likely values: guess - 1, guess, guess + 1
		if (key < arr[guess]) {
			if (key < arr[guess - 1]) {
				imax = guess - 1;
				// last attempt to restrict search to items in cache
				if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
					imin = guess - LIKELY_IN_CACHE_SIZE;
				}
			} else {
				// key >= arr[guess - 1]
				return guess - 1;
			}
		} else {
			// key >= arr[guess]
			if (key < arr[guess + 1]) {
				return guess;
			} else {
				// key >= arr[guess + 1]
				if (key < arr[guess + 2]) {
					return guess + 1;
				} else {
					// key >= arr[guess + 2]
					imin = guess + 2;
					// last attempt to restrict search to items in cache
					if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
						imax = guess + LIKELY_IN_CACHE_SIZE;
					}
				}
			}
		}
		// finally, find index by bisection
		while (imin < imax) {
			int imid = imin + ((imax - imin) >> 1);
			if (key >= arr[imid]) {
				imin = imid + 1;
			} else {
				imax = imid;
			}
		}
		return imin - 1;
	}

	private static int linearSearch(double key, double[] arr, int len, int i0) {
		int i;
		for (i = i0; i < len && key >= arr[i]; i++)
			;
		return i - 1;
	}

	private static int binarySearchWithGuessReverseNegated(double key, double[] arr, int len, int guess, int lenminusone) {
		int imin = 0;
		int imax = len;
		// Handle keys outside of the arr range first
		if (key > -arr[lenminusone - (len - 1)]) {
			return len;
		} else if (key < -arr[lenminusone]) {
			return -1;
		}
		// If len <= 4 use linear search. From above we know key >= arr[0] when we start.
		if (len <= 4) {
			return linearSearchReverseNegated(key, arr, len, 1, lenminusone);
		}
		if (guess > len - 3) {
			guess = len - 3;
		}
		if (guess < 1) {
			guess = 1;
		}
		// check most likely values: guess - 1, guess, guess + 1
		if (key < -arr[lenminusone - guess]) {
			if (key < -arr[lenminusone - (guess - 1)]) {
				imax = guess - 1;
				// last attempt to restrict search to items in cache
				if (guess > LIKELY_IN_CACHE_SIZE && key >= -arr[lenminusone - (guess - LIKELY_IN_CACHE_SIZE)]) {
					imin = guess - LIKELY_IN_CACHE_SIZE;
				}
			} else {
				// key >= arr[guess - 1]
				return guess - 1;
			}
		} else {
			// key >= arr[guess]
			if (key < -arr[lenminusone - (guess + 1)]) {
				return guess;
			} else {
				// key >= arr[guess + 1]
				if (key < -arr[lenminusone - (guess + 2)]) {
					return guess + 1;
				} else {
					// key >= arr[guess + 2]
					imin = guess + 2;
					// last attempt to restrict search to items in cache
					if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < -arr[lenminusone - (guess + LIKELY_IN_CACHE_SIZE)]) {
						imax = guess + LIKELY_IN_CACHE_SIZE;
					}
				}
			}
		}
		// finally, find index by bisection
		while (imin < imax) {
			int imid = imin + ((imax - imin) >> 1);
			if (key >= -arr[lenminusone - imid]) {
				imin = imid + 1;
			} else {
				imax = imid;
			}
		}
		return imin - 1;
	}

	private static int linearSearchReverseNegated(double key, double[] arr, int len, int i0, int lenminusone) {
		int i;
		for (i = i0; i < len && key >= -arr[lenminusone - i]; i++)
			;
		return i - 1;
	}

	/**
	 * if x[i] == val_x then set y[i] == val_y
	 * 
	 * @param x
	 * @param lowerBoundX
	 * @param out
	 * @param LOWER_BOUND_Y
	 */
	private static void eqApply(double[] x, double val_x, double[] y, double val_y) {
		int len = x.length;
		for (int i = 0; i < len; i++) {
			if (x[i] == val_x) {
				y[i] = val_y;
			}
		}
	}

	@Override
	public int getColInputCount() {
		// Currently the implementation only supports one dimensional inputs/outputs
		return 1;
	}

	@Override
	public int getColOutputCount() {
		// Currently the implementation only supports one dimensional inputs/outputs
		return 1;
	}

	@Override
	public void shutdown() {
		// DO NOT REMOVE REFERENCES TO INTERNAL OBJECTS HERE SINCE CACHED COPIES OF THE TRANSFORMER
		// MIGHT BE SHARED
	}

	@Override
	public int compareTo(QuantileTransformer other) {
		if (nQuantiles != other.nQuantiles) {
			return -1;
		}
		if (subsample != other.subsample) {
			return -1;
		}
		if (lowerBoundX != other.lowerBoundX) {
			return -1;
		}
		if (upperBoundX != other.upperBoundX) {
			return -1;
		}
		if (!Arrays.equals(references, other.references)) {
			return -1;
		}
		if (!Arrays.equals(quantiles, other.quantiles)) {
			return -1;
		}
		return 0;
	}

}
