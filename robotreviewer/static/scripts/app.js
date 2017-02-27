/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';

  var Backbone = require("backbone");
  var React = require("react");
  var ReactDOM = require("react-dom");
  var _ = require("underscore");
  var $ = require("jquery");
  var FileUtil = require("spa/helpers/fileUtil");


  var Fastclick = require("fastclick");
  var Foundation = require("foundation");


  // Set CSRF
  var _sync = Backbone.sync;
  Backbone.sync = function(method, model, options){
    options.beforeSend = function(xhr){
      xhr.setRequestHeader('X-CSRF-Token', window.CSRF_TOKEN);
    };
    return _sync(method, model, options);
  };

  // Breadcrumbs hack
  var breadcrumbsModel = new (require("models/breadcrumbs"))();
  var BreadcrumbsComponent = React.createFactory(require("jsx!components/breadcrumbs"));

  var breadcrumbs =
      ReactDOM.render(new BreadcrumbsComponent({breadcrumbs: breadcrumbsModel}),
                                    document.getElementById("breadcrumbs"));

  breadcrumbsModel.on("all", function(e, obj) {
    breadcrumbs.forceUpdate();
  });

  // Component views
  var DocumentView = React.createFactory(require("jsx!views/document"));
  var UploadView = React.createFactory(require("jsx!views/upload"));
  var ReportView = React.createFactory(require("jsx!views/report"));

  var isEditable = false;

  var Router = Backbone.Router.extend({
    routes : {
      "upload" : "upload",
      "report/:reportId" : "report",
      "document/:reportId/:documentId?annotation_type=:type&uuid=:uuid" : "document",
      "document/:reportId/:documentId?annotation_type=:type" : "document",
      "*path" : "upload"
    },
    upload : function() {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(new UploadView({}), node);
      breadcrumbsModel.reset(
        [{link: "/#upload", title: "upload"}]);
    },
    report : function(reportId) {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);
      ReactDOM.render(new ReportView({reportId: reportId}), node);
      breadcrumbsModel.reset(
        [{link: "/#upload", title: "upload"},
         {link: "/#report/" + reportId, title: "report"}
        ]);
    },
    document : function(reportId, documentId, type, uuid) {
      var node = document.getElementById("main");
      ReactDOM.unmountComponentAtNode(node);

      // Models
      var documentModel = new (require("spa/models/document"))();
      var marginaliaModel = new (require("spa/models/marginalia"))();

      var marginaliaUrl = "/marginalia/" + reportId + "/" + documentId + "?annotation_type=" + type;
      $.get(marginaliaUrl, function(data) {
        var marginalia = {marginalia: JSON.parse(data)};
        marginaliaModel.reset(marginaliaModel.parse(marginalia));
        if(uuid) {
          marginaliaModel.setActiveByUuid(uuid);
        }
      });

      var documentUrl = "/pdf/" + reportId + "/" + documentId;
      documentModel.loadFromUrl(documentUrl, uuid);

      ReactDOM.render(
        new DocumentView({document: documentModel,
                          marginalia: marginaliaModel,
                          isEditable: isEditable}),
        node);

      breadcrumbsModel.reset(
        [{link: "/#upload", title: "upload"},
         {link: "/#report/" + reportId, title: "report"},
         {link: "/#document/" + reportId + "/" + documentId, title: "document"}
        ]);
    }
  });

  window.router = new Router();


  Backbone.history.start();

  $(document).ready(function() {
    $(document).foundation();
  })
});
