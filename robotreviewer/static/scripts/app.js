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
  var TopBar = React.createFactory(require("jsx!components/topBar"));
  var DocumentView = React.createFactory(require("jsx!views/document"));
  var ReportView = React.createFactory(require("jsx!views/report"));


  var process = function(data) {
    //var upload = FileUtil.upload("/topologies/ebm", data);
    documentModel.loadFromData(data);
    // upload.then(function(result) {
    //  var marginalia = JSON.parse(result);
    //  marginaliaModel.reset(marginaliaModel.parse(marginalia));
    //});
  };

  var topBarComponent = ReactDOM.render(
    new TopBar({
      callback: process,
      accept: ".pdf",
      mimeType: /application\/(x-)?pdf|text\/pdf/
    }),
    document.getElementById("top-bar")
  );

  var isEditable = true;

  var Router = Backbone.Router.extend({
    routes : {
      "report"   : "report",
      "document" : "document",
      "*path"    : "report"
    },
    report : function() {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(new ReportView({}), node);
    },
    document : function() {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(
        new DocumentView({document: documentModel, marginalia: marginaliaModel, isEditable: isEditable}),
        node);
    }
  });

  new Router();


  Backbone.history.start();
});
