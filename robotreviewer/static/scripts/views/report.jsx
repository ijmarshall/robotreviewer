/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';
  var React = require("react");

  var Dropzone = require("react-dropzone")
  var _ = require("underscore");
  var $ = require("jquery");

  var ReportView = React.createClass({
    componentDidMount: function() {
      var $frame = $(this.refs.frame);
      var html = $.get("report_view/html", function(data) {
        $frame.contents().find("body").html(data);
      });
    },
    render: function() {
      return(<div className="report"><iframe ref="frame" /></div>);
    }
  });

  return ReportView;
});
