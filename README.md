# quantile-transformer
A Quantile Transformer in Java, adapted from scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html">sklearn.preprocessing.QuantileTransformer</a>.

Dependencies
-
The transformer code has some logging requiring org.apache.log4j.Logger.
The transformer test code uses <a href="https://github.com/bcdev/jpy">jpy</a> to compare results directly with Python. It requires the system property "jpy.jpyLib" to be set.

Limitations
-
<b>NOTE:</b> This implementation only supports <code>subsample</code> == <code>nSamples</code> and will throw an <code>UnsupportedOperationException</code> otherwise.<br/>
Additionally, this implementation only supports a single input feature/column.
