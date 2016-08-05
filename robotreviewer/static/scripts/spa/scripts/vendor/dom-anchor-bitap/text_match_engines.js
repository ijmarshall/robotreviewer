define(function(require, exports, module) {
  var DMP, DMPMatcher, ExactMatcher, RegexMatcher;

  DMP = require("./vendor/diff-match-patch/diff_match_patch_uncompressed");

  ExactMatcher = (function() {
    function ExactMatcher() {
      this.distinct = true;
      this.caseSensitive = false;
    }

    ExactMatcher.prototype.setDistinct = function(value) {
      return this.distinct = value;
    };

    ExactMatcher.prototype.setCaseSensitive = function(value) {
      return this.caseSensitive = value;
    };

    ExactMatcher.prototype.search = function(text, pattern) {
      var i, index, pLen, results;
      pLen = pattern.length;
      results = [];
      index = 0;
      if (!this.caseSensitive) {
        text = text.toLowerCase();
        pattern = pattern.toLowerCase();
      }
      while ((i = text.indexOf(pattern)) > -1) {
        (function(_this) {
          return (function() {
            results.push({
              start: index + i,
              end: index + i + pLen
            });
            if (_this.distinct) {
              text = text.substr(i + pLen);
              return index += i + pLen;
            } else {
              text = text.substr(i + 1);
              return index += i + 1;
            }
          });
        })(this)();
      }
      return results;
    };

    return ExactMatcher;

  })();

  RegexMatcher = (function() {
    function RegexMatcher() {
      this.caseSensitive = false;
    }

    RegexMatcher.prototype.setCaseSensitive = function(value) {
      return this.caseSensitive = value;
    };

    RegexMatcher.prototype.search = function(text, pattern) {
      var m, re, _results;
      re = new RegExp(pattern, this.caseSensitive ? "g" : "gi");
      _results = [];
      while (m = re.exec(text)) {
        _results.push({
          start: m.index,
          end: m.index + m[0].length
        });
      }
      return _results;
    };

    return RegexMatcher;

  })();

  DMPMatcher = (function() {
    function DMPMatcher() {
      this.dmp = new DMP.diff_match_patch();
      this.dmp.Diff_Timeout = 0;
      this.caseSensitive = false;
    }

    DMPMatcher.prototype._reverse = function(text) {
      return text.split("").reverse().join("");
    };

    DMPMatcher.prototype.getMaxPatternLength = function() {
      return this.dmp.Match_MaxBits;
    };

    DMPMatcher.prototype.setMatchDistance = function(distance) {
      return this.dmp.Match_Distance = distance;
    };

    DMPMatcher.prototype.getMatchDistance = function() {
      return this.dmp.Match_Distance;
    };

    DMPMatcher.prototype.setMatchThreshold = function(threshold) {
      return this.dmp.Match_Threshold = threshold;
    };

    DMPMatcher.prototype.getMatchThreshold = function() {
      return this.dmp.Match_Threshold;
    };

    DMPMatcher.prototype.getCaseSensitive = function() {
      return caseSensitive;
    };

    DMPMatcher.prototype.setCaseSensitive = function(value) {
      return this.caseSensitive = value;
    };

    DMPMatcher.prototype.search = function(text, pattern, expectedStartLoc, options) {
      var endIndex, endLen, endLoc, endPos, endSlice, found, matchLen, maxLen, pLen, result, startIndex, startLen, startPos, startSlice;
      if (expectedStartLoc == null) {
        expectedStartLoc = 0;
      }
      if (options == null) {
        options = {};
      }
      if (expectedStartLoc < 0) {
        throw new Error("Can't search at negative indices!");
      }
      if (expectedStartLoc !== Math.floor(expectedStartLoc)) {
        throw new Error("Expected start location must be an integer.");
      }
      if (!this.caseSensitive) {
        text = text.toLowerCase();
        pattern = pattern.toLowerCase();
      }
      pLen = pattern.length;
      maxLen = this.getMaxPatternLength();
      if (pLen <= maxLen) {
        result = this.searchForSlice(text, pattern, expectedStartLoc);
      } else {
        startSlice = pattern.substr(0, maxLen);
        startPos = this.searchForSlice(text, startSlice, expectedStartLoc);
        if (startPos != null) {
          startLen = startPos.end - startPos.start;
          endSlice = pattern.substr(pLen - maxLen, maxLen);
          endLoc = startPos.start + pLen - maxLen;
          endPos = this.searchForSlice(text, endSlice, endLoc);
          if (endPos != null) {
            endLen = endPos.end - endPos.start;
            matchLen = endPos.end - startPos.start;
            startIndex = startPos.start;
            endIndex = endPos.end;
            if ((pLen * 0.5 <= matchLen && matchLen <= pLen * 1.5)) {
              result = {
                start: startIndex,
                end: endPos.end
              };
            }
          }
        }
      }
      if (result == null) {
        return [];
      }
      if (options.withLevenhstein || options.withDiff) {
        found = text.substr(result.start, result.end - result.start);
        result.diff = this.dmp.diff_main(pattern, found);
        if (options.withLevenshstein) {
          result.lev = this.dmp.diff_levenshtein(result.diff);
        }
        if (options.withDiff) {
          this.dmp.diff_cleanupSemantic(result.diff);
          result.diffHTML = this.dmp.diff_prettyHtml(result.diff);
        }
      }
      return [result];
    };

    DMPMatcher.prototype.compare = function(text1, text2, explain) {
      var c, changes, result, _i, _len, _ref;
      if (explain == null) {
        explain = false;
      }
      if (!((text1 != null) && (text2 != null))) {
        throw new Error("Can not compare non-existing strings!");
      }
      result = {};
      result.diff = this.dmp.diff_main(text1, text2);
      result.lev = this.dmp.diff_levenshtein(result.diff);
      result.errorLevel = result.lev / text1.length;
      this.dmp.diff_cleanupSemantic(result.diff);
      result.diffHTML = this.dmp.diff_prettyHtml(result.diff);
      if (explain) {
        changes = [];
        _ref = result.diff;
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          c = _ref[_i];
          switch (c[0]) {
            case -1:
              changes.push("-'" + c[1] + "'");
              break;
            case +1:
              changes.push("+'" + c[1] + "'");
          }
        }
        result.diffExplanation = changes.join(", ");
      }
      return result;
    };

    DMPMatcher.prototype.searchForSlice = function(text, slice, expectedStartLoc) {
      var dneIndex, endIndex, expectedDneLoc, expectedEndLoc, nrettap, r1, r2, result, startIndex, txet;
      r1 = this.dmp.match_main(text, slice, expectedStartLoc);
      startIndex = r1.index;
      if (startIndex === -1) {
        return null;
      }
      txet = this._reverse(text);
      nrettap = this._reverse(slice);
      expectedEndLoc = startIndex + slice.length;
      expectedDneLoc = text.length - expectedEndLoc;
      r2 = this.dmp.match_main(txet, nrettap, expectedDneLoc);
      dneIndex = r2.index;
      endIndex = text.length - dneIndex;
      return result = {
        start: startIndex,
        end: endIndex
      };
    };

    return DMPMatcher;

  })();

  module.exports = {
    exact: ExactMatcher,
    regex: RegexMatcher,
    dmp: DMPMatcher
  };

});
