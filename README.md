# quantile-transformer
A Quantile Transformer in Java, adapted from scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html">sklearn.preprocessing.QuantileTransformer</a>.

Use
-
```
// Some train data
double[] x = new double[] { 0.4854325053825786, 0.4645123710572827, 0.5446322647838966, 0.4660098290732253, 0.50266833449657,
    0.5050523348417824, 0.5015153877587296, 0.4498840370128503, 0.4728002823881743, 0.4708978286126443 };

// Create a new transformer
QuantileTransformer qt = new QuantileTransformer(x.length, x.length);

// Fit the transformer to the data
qt.fit(x);

// Get the quantiles
double[] x_transformed = aqt.transform(x, double[].class, true);

// Shutdown
qt.shutdown();
```

Dependencies
-
The transformer code has some logging requiring org.apache.log4j.Logger.
The transformer test code uses <a href="https://github.com/bcdev/jpy">jpy</a> to compare results directly with Python. It requires the system property "jpy.jpyLib" to be set.

Limitations
-
<b>NOTE:</b> This implementation only supports <code>subsample</code> == <code>nSamples</code> and will throw an <code>UnsupportedOperationException</code> otherwise.<br/>
Additionally, this implementation only supports a single input feature/column.
