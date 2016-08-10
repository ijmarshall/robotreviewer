/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");
  var _ = require("underscore");

  var Document = require("jsx!../spa/components/document");
  var Marginalia = require("jsx!../spa/components/marginalia");


  var DocumentView = React.createClass({
    componentDidMount: function() {
      var self = this;
      var marginaliaModel = this.props.marginalia;
      var documentModel = this.props.document;

      // Dispatch logic
      // Listen to model change callbacks -> trigger updates to components
      marginaliaModel.on("all", function(e, obj) {
        switch(e) {
        case "reset":
          documentModel.annotate(marginaliaModel.getActive());
          break;
        case "annotations:change":
          break;
        case "change:active":
        case "annotations:add":
        case "annotations:remove":
          documentModel.annotate(marginaliaModel.getActive());
          self.forceUpdate();
          break;
        case "annotations:select":
          break;
        default:
          break;
        }
      });

      documentModel.on("all", function(e, obj) {
        switch(e) {
        case "change:raw":
          self.forceUpdate();
          break;
        case "change:binary":
          marginaliaModel.reset();
          break;
        case "pages:change:state":
          if(obj.get("state") === window.RenderingStates.HAS_CONTENT) {
            documentModel.annotate(marginaliaModel.getActive());
          }
          self.forceUpdate();
          break;
        case "pages:ready":
          documentModel.annotate(marginaliaModel.getActive());
          self.forceUpdate();
          break;
        case "pages:change:annotations":
          documentModel.annotate(marginaliaModel.getActive());
          self.forceUpdate();
          break;
        default:
          break;
        }
      });
    },
    componentWillUnmount: function() {
      var marginaliaModel = this.props.marginalia;
      var documentModel = this.props.document;

      marginaliaModel.off("all");
      documentModel.off("all");
    },
    render: function() {
      var self = this;
      var marginaliaModel = this.props.marginalia;
      var documentModel = this.props.document;
      var isEditable = this.props.isEditable;

      return(
          <div>
            <Document id="viewer" pdf={documentModel} marginalia={marginaliaModel} isEditable={isEditable} />
            <div id="side">
              <Marginalia marginalia={marginaliaModel} isEditable={isEditable} />
            </div>
          </div>
      );
    }
  });

  return DocumentView;
});
