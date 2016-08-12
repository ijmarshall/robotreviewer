/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';

  var _ = require("underscore");
  var Backbone = require("backbone");

  var Breadcrumb = Backbone.Model.extend({
    defaults: {
      link: null,
      title: null
    }
  });

  var Breadcrumbs = Backbone.Collection.extend({
    model: Breadcrumb
  });

  return Breadcrumbs;
});
