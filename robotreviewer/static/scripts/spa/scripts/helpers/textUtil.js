/* -*- mode: js2; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js2-basic-offset: 2 -*- */
define(function (require) {
  'use strict';

  var TextUtil = function textUtil() {
    var NORMALIZE_PATTERN = /(\r\n|\n|\r|\s{2,})/g;

    this.normalize = function(str) {
      return str.trim().replace(NORMALIZE_PATTERN," ");
    };
  };

  return new TextUtil();
});
