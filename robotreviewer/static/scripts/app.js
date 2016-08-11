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

  var isEditable = false;

  var Router = Backbone.Router.extend({
    routes : {
      "upload" : "upload",
      "report/:reportId" : "report",
      "document/:reportId/:documentId?annotation_type=:type" : "document",
      "*path" : "upload"
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
    document : function(reportId, documentId, type) {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);

      documentModel.set({binary: null, _cache: {}});

      var marginaliaUrl = "/marginalia/" + reportId + "/" + documentId + "?annotation_type=" + type;
      $.get(marginaliaUrl, function(data) {
        var marginalia = {marginalia: JSON.parse(data)};
        marginaliaModel.reset(marginaliaModel.parse(marginalia));
      });

      var documentUrl = "/pdf/" + reportId + "/" + documentId;
      documentModel.loadFromUrl(documentUrl);

      ReactDOM.render(
        new DocumentView({document: documentModel, marginalia: marginaliaModel, isEditable: isEditable}),
        node);
    }
  });

  window.router = new Router();


  Backbone.history.start();
});
