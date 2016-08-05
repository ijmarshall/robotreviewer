This is a fork of Google's "Diff, Match and Patch Library", maintained
by the Hypothes.is project.

(See README.upstream.txt for upstream information.)

We created this fork because we needed some enhanced functionality in
the JS version.

Changes:
 * Modified match_main to return more data about the matches.
   This was needed for implementing searches for longer text segments,
   which are done by searching for multiple smaller segments,
   and aggregating the results.
   To aggregate the results, we need the individual lev. distances
   of the smaller matches, which the original version did not provide.

 * Added a workaround for [Chrome bug #2790](https://code.google.com/p/v8/issues/detail?id=2790).
