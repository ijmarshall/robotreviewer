define(function(require, exports, module) {
  var TextSearcher, engines;

  engines = require("./text_match_engines");

  TextSearcher = (function() {
    function TextSearcher() {}

    TextSearcher.prototype.searchExact = function(corpus, pattern, distinct, caseSensitive) {
      if (distinct == null) {
        distinct = true;
      }
      if (caseSensitive == null) {
        caseSensitive = false;
      }
      if (this.pm == null) {
        this.pm = new engines.exact;
      }
      this.pm.setDistinct(distinct);
      this.pm.setCaseSensitive(caseSensitive);
      return this._search(corpus, this.pm, pattern);
    };

    TextSearcher.prototype.searchRegex = function(corpus, pattern, caseSensitive) {
      if (caseSensitive == null) {
        caseSensitive = false;
      }
      if (this.rm == null) {
        this.rm = new engines.regex;
      }
      this.rm.setCaseSensitive(caseSensitive);
      return this._search(corpus, this.rm, pattern);
    };

    TextSearcher.prototype.searchFuzzy = function(corpus, pattern, pos, caseSensitive, options) {
      var _ref, _ref1;
      if (caseSensitive == null) {
        caseSensitive = false;
      }
      if (options == null) {
        options = {};
      }
      this.ensureDMP();
      this.dmp.setMatchDistance((_ref = options.matchDistance) != null ? _ref : 1000);
      this.dmp.setMatchThreshold((_ref1 = options.matchThreshold) != null ? _ref1 : 0.5);
      this.dmp.setCaseSensitive(caseSensitive);
      return this._search(corpus, this.dmp, pattern, pos, options);
    };

    TextSearcher.prototype.searchFuzzyWithContext = function(corpus, prefix, suffix, pattern, expectedStart, expectedEnd, caseSensitive, options) {
      var a1, a2, analysis, bestError, candidate, charRange, expectedPrefixStart, expectedSuffixStart, flexMatch, flexMatches, flexRange, i, k, len, match, matchThreshold, obj, patternLength, prefixEnd, prefixError, prefixRange, prefixResult, prefixStart, remainingText, suffixEnd, suffixError, suffixRange, suffixResult, suffixStart, totalError, v, _i, _j, _len, _len1, _ref, _ref1, _ref2, _ref3;
      if (expectedStart == null) {
        expectedStart = null;
      }
      if (expectedEnd == null) {
        expectedEnd = null;
      }
      if (caseSensitive == null) {
        caseSensitive = false;
      }
      if (options == null) {
        options = {};
      }
      this.ensureDMP();
      if (!((prefix != null) && (suffix != null))) {
        throw new Error("Can not do a context-based fuzzy search with missing context!");
      }
      len = corpus.length;
      expectedPrefixStart = expectedStart != null ? (i = expectedStart - prefix.length, i < 0 ? 0 : i) : Math.floor(len / 2);
      this.dmp.setMatchDistance((_ref = options.contextMatchDistance) != null ? _ref : len * 2);
      this.dmp.setMatchThreshold((_ref1 = options.contextMatchThreshold) != null ? _ref1 : 0.5);
      prefixResult = this.dmp.search(corpus, prefix, expectedPrefixStart);
      if (!prefixResult.length) {
        return {
          matches: []
        };
      }
      prefixStart = prefixResult[0].start;
      prefixEnd = prefixResult[0].end;
      patternLength = pattern != null ? pattern.length : (expectedStart != null) && (expectedEnd != null) ? expectedEnd - expectedStart : 64;
      remainingText = corpus.substr(prefixEnd);
      expectedSuffixStart = patternLength;
      suffixResult = this.dmp.search(remainingText, suffix, expectedSuffixStart);
      if (!suffixResult.length) {
        return {
          matches: []
        };
      }
      suffixStart = prefixEnd + suffixResult[0].start;
      suffixEnd = prefixEnd + suffixResult[0].end;
      charRange = {
        start: prefixEnd,
        end: suffixStart
      };
      matchThreshold = (_ref2 = options.patternMatchThreshold) != null ? _ref2 : 0.5;
      analysis = this._analyzeMatch(corpus, pattern, charRange, true);
      if ((pattern != null) && options.flexContext && !analysis.exact) {
        if (this.pm == null) {
          this.pm = new engines.exact;
        }
        this.pm.setDistinct(false);
        this.pm.setCaseSensitive(false);
        flexMatches = this.pm.search(corpus.slice(prefixStart, suffixEnd), pattern);
        delete candidate;
        bestError = 2;
        for (_i = 0, _len = flexMatches.length; _i < _len; _i++) {
          flexMatch = flexMatches[_i];
          flexRange = {
            start: prefixStart + flexMatch.start,
            end: prefixStart + flexMatch.end
          };
          prefixRange = {
            start: prefixStart,
            end: flexRange.start
          };
          a1 = this._analyzeMatch(corpus, prefix, prefixRange, true);
          prefixError = a1.exact ? 0 : a1.comparison.errorLevel;
          suffixRange = {
            start: flexRange.end,
            end: suffixEnd
          };
          a2 = this._analyzeMatch(corpus, suffix, suffixRange, true);
          suffixError = a2.exact ? 0 : a2.comparison.errorLevel;
          if (a1.exact || a2.exact) {
            totalError = prefixError + suffixError;
            if (totalError < bestError) {
              candidate = flexRange;
              bestError = totalError;
            }
          }
        }
        if (candidate != null) {
          console.log("flexContext adjustment: we found a better candidate!");
          charRange = candidate;
          analysis = this._analyzeMatch(corpus, pattern, charRange, true);
        }
      }
      if ((pattern == null) || analysis.exact || (analysis.comparison.errorLevel <= matchThreshold)) {
        match = {};
        _ref3 = [charRange, analysis];
        for (_j = 0, _len1 = _ref3.length; _j < _len1; _j++) {
          obj = _ref3[_j];
          for (k in obj) {
            v = obj[k];
            match[k] = v;
          }
        }
        return {
          matches: [match]
        };
      }
      return {
        matches: []
      };
    };

    TextSearcher.prototype._normalizeString = function(string) {
      return (string.replace(/\s{2,}/g, " ")).trim();
    };

    TextSearcher.prototype._search = function(corpus, matcher, pattern, pos, options) {
      var fuzzyComparison, matches, result, t1, t2, t3, textMatch, textMatches, _fn, _i, _len, _ref;
      if (options == null) {
        options = {};
      }
      if (pattern == null) {
        throw new Error("Can't search for null pattern!");
      }
      pattern = pattern.trim();
      if (pattern == null) {
        throw new Error("Can't search an for empty pattern!");
      }
      fuzzyComparison = (_ref = options.withFuzzyComparison) != null ? _ref : false;
      t1 = this.timestamp();
      textMatches = matcher.search(corpus, pattern, pos, options);
      t2 = this.timestamp();
      matches = [];
      _fn = (function(_this) {
        return function(textMatch) {
          var analysis, k, match, obj, v, _j, _len1, _ref1;
          analysis = _this._analyzeMatch(corpus, pattern, textMatch, fuzzyComparison);
          match = {};
          _ref1 = [textMatch, analysis];
          for (_j = 0, _len1 = _ref1.length; _j < _len1; _j++) {
            obj = _ref1[_j];
            for (k in obj) {
              v = obj[k];
              match[k] = v;
            }
          }
          matches.push(match);
          return null;
        };
      })(this);
      for (_i = 0, _len = textMatches.length; _i < _len; _i++) {
        textMatch = textMatches[_i];
        _fn(textMatch);
      }
      t3 = this.timestamp();
      result = {
        matches: matches,
        time: {
          phase1_textMatching: t2 - t1,
          phase2_matchMapping: t3 - t2,
          total: t3 - t1
        }
      };
      return result;
    };

    TextSearcher.prototype.timestamp = function() {
      return new Date().getTime();
    };

    TextSearcher.prototype._analyzeMatch = function(corpus, pattern, charRange, useFuzzy) {
      var expected, found, result;
      if (useFuzzy == null) {
        useFuzzy = false;
      }
      expected = this._normalizeString(pattern);
      found = this._normalizeString(corpus.slice(charRange.start, charRange.end));
      result = {
        found: found,
        exact: found === expected
      };
      if (!result.exact) {
        result.exactExceptCase = expected.toLowerCase() === found.toLowerCase();
      }
      if (!result.exact && useFuzzy) {
        this.ensureDMP();
        result.comparison = this.dmp.compare(expected, found);
      }
      return result;
    };

    TextSearcher.prototype.ensureDMP = function() {
      return this.dmp != null ? this.dmp : this.dmp = new engines.dmp;
    };

    return TextSearcher;

  })();

  module.exports = TextSearcher;

});
