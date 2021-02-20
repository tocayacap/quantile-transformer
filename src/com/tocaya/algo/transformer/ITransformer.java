package com.tocaya.algo.transformer;

/**
 * A simple interface for data transformers.
 */
public interface ITransformer {

	/**
	 * Fit the transformer using the input data.
	 * 
	 * @param x
	 *            Input data
	 */
	public <I> void fit(I x);

	/**
	 * The transform method allows for different input/output types in order to provide flexibility
	 * for the implementation.
	 * 
	 * @param x
	 *            Input data
	 * @param returnType
	 *            The type of the data returned by the transformer
	 * @param rowMajorInput
	 *            If the input is provided in "row major" format. This setting is applicable when
	 *            the input data type is an array-like data structure.
	 * @return Transformed data
	 */
	public <I, O> O transform(I x, Class<O> returnType, boolean rowMajorInput);

	/**
	 * @return The number of "columns" in the input. The definition of a column is determined
	 *         according to the implementation.
	 */
	public int getColInputCount();

	/**
	 * @return The number of "columns" in the output. The definition of a column is determined
	 *         according to the implementation.
	 */
	public int getColOutputCount();

	/**
	 * Shutdown any resources used by this transformer.
	 */
	public void shutdown();

}
