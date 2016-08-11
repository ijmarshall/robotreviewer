/* -*- mode: js2; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js2-basic-offset: 2 -*- */
define(function (require) {
  'use strict';

  var _ = require("underscore");
  var $ = require("jquery");
  var React = require("react");

  var Annotate = require("jsx!./annotate");
  var Minimap = require("jsx!./minimap");
  var Page = require("jsx!./page");
  var TextUtil = require("../helpers/textUtil");

  var Immutable = require("immutable");

  var PDFJS = require("PDFJS");
  var PDFJSUrl = require.toUrl('PDFJS');

  PDFJS.cMapUrl = PDFJSUrl.replace(/\/pdf$/, '') + '/cmaps/';
  PDFJS.cMapPacked = true;
  PDFJS.disableWebGL = false;

  PDFJS.workerSrc = PDFJSUrl + ".worker.js";

  var Document = React.createClass({
    getInitialState: function() {
      return { $viewer: null };
    },
    toggleHighlights: function(e, uuid) {
      var $annotations = this.state.$viewer.find("[data-uuid*="+uuid+"]");
      $annotations.toggleClass("highlight");
    },
    scrollTo: function(uuid) {
      var $viewer = this.state.$viewer;
      if($viewer) {
        var annotation = $viewer.find("[data-uuid*="+ uuid + "]")
        if(annotation) {
          var delta = annotation.offset().top;
          var viewerHeight = $viewer.height();
          var center = viewerHeight / 2;
          $viewer.animate({scrollTop: $viewer.scrollTop() + delta - center});
        }
      }
    },
    componentWillUnmount: function() {
      $(window).off("highlight", this.toggleHighlights);
      this.props.marginalia.off("annotations:select", this.scrollTo);
    },
    componentDidMount: function() {
      $(window).on("highlight", this.toggleHighlights);
      this.props.marginalia.on("annotations:select", this.scrollTo);

      var $viewer = $(this.refs.viewer);
      this.setState({$viewer: $viewer});
    },
    render: function() {
      var pdf = this.props.pdf;
      var marginalia = this.props.marginalia;

      var fingerprint = pdf.get("fingerprint");
      var pages = pdf.get("pages");

      var annotations = Immutable.fromJS(pages.map(function(page, index) {
        return page.get("annotations");
      }));

      var pagesElements = pdf.get("pages").map(function(page, pageIndex) {
        return (<Page page={page} key={fingerprint + pageIndex} annotations={annotations.get(pageIndex)} />);
      });

      return(
        <div>
          <Minimap $viewer={this.state.$viewer} pdf={pdf} annotations={annotations} />
          <div className="viewer-container">
            <div className="viewer" ref="viewer">
               {this.props.isEditable ? <Annotate marginalia={marginalia} /> : null}
               {pagesElements}
             </div>
           </div>
        </div>);
    }
  });

  return Document;
});
