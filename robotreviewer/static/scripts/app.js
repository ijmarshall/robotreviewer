/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';

  var Backbone = require("backbone");
  var React = require("react");
  var ReactDOM = require("react-dom");
  var _ = require("underscore");
  var FileUtil = require("spa/helpers/fileUtil");

  // Set CSRF
  var _sync = Backbone.sync;
  Backbone.sync = function(method, model, options){
    options.beforeSend = function(xhr){
      xhr.setRequestHeader('X-CSRF-Token', window.CSRF_TOKEN);
    };
    return _sync(method, model, options);
  };

  // Models
  var documentModel = new (require("spa/models/document"))();
  var marginaliaModel = new (require("spa/models/marginalia"))();

  // Components
  var DocumentView = React.createFactory(require("jsx!views/document"));
  var UploadView = React.createFactory(require("jsx!views/upload"));
  var ReportView = React.createFactory(require("jsx!views/report"));

  var isEditable = true;

  var Router = Backbone.Router.extend({
    routes : {
      "upload"              : "upload",
      "report/:reportId"    : "report",
      "document"            : "document",
      "*path"               : "upload"
    },
    upload : function() {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(new UploadView({}), node);
    },
    report : function(reportId) {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(new ReportView({reportId: reportId}), node);
    },
    document : function() {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(
        new DocumentView({document: documentModel, marginalia: marginaliaModel, isEditable: isEditable}),
        node);
    }
  });

  window.router = new Router();


  Backbone.history.start();
});
