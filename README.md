# sddr

a repository holding tools to estimate the Savage-Dickey Density Ratio (sddr) specifically for a one-dimensional posterior.
we do this with the following methods

  * raw histogram (counts)
  * cumulative histogram fit to a low-order polynomial (slope)
  * kde with reflecting boundary condition at the prior bounds, marginalized over the kde bandwidth

The repo supports plotting & comparison tools to sanity check by eye.
It also supports distributions of our estimates of the sddr
