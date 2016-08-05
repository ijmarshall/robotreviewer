/* -*- mode: js2; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js2-basic-offset: 2 -*- */
define(function (require) {
  'use strict';

  var React = require("react");
  var $ = require("jquery");

  var textSelected = function() {
    return /(\w{2,}\W{1,6}){3}/.test(window.document.getSelection().toString());
  };

  var getSelection = function() {
    var selection = window.document.getSelection();
    if(selection.type === "None" || !selection.getRangeAt(0)) return "";
    var range = selection.getRangeAt(0);
    var strArr = [];
    var childNodes = range.cloneContents().childNodes;
    for (var i = 0, len = childNodes.length; i < len; i++) {
      strArr.push(childNodes[i].textContent);
    }
    return strArr.join(" ").trim();
  };

  var timeout = null;

  var Annotate = React.createClass({
    getInitialState: function() {
      return { visible: false };
    },
    componentWillUnmount: function() {
      var $container = $(this.refs.popup.getDOMNode()).parent();
      $container.off("mouseup.popup");
    },
    componentDidMount: function() {
      var self = this;
      var $container = $(this.refs.popup.getDOMNode()).parent();

      $container.on("mouseup.popup", function(e) {
        window.clearTimeout(timeout);
        self.hide();

        timeout = window.setTimeout(function() {
          if(textSelected() && !$(e.target).is(":input")) {
            self.show(e.pageX);
          }
        }, 200);
      });

    },
    show: function(mouseX)  {
      if(this.state.visible) return;

      var $popup = $(this.refs.popup.getDOMNode());
      var $container = $popup.parent();

      var selectionBox = window.document.getSelection().getRangeAt(0).getBoundingClientRect();
      var selectionLeft = selectionBox.left + $container.scrollLeft() - $container.offset().left;
      var selectionTop = selectionBox.top + $container.scrollTop() - $container.offset().top;
      var buttonWidth = $popup.outerWidth();

      var left, top;
      if(mouseX) {
        left = Math.min(Math.max(mouseX, selectionLeft + (buttonWidth/2)),
                        selectionLeft + selectionBox.width - (buttonWidth/2));
      } else {
        left = selectionLeft + (selectionBox.width/2);
      }

      left = left - (buttonWidth/2);
      top = selectionTop - 2 - $popup.outerHeight();

      this.setState(
        {visible: true,
         top: top,
         left: left}
      );
    },
    hide: function() {
      this.setState({visible: false});
    },
    _down: function(e) {
      e.preventDefault();
    },
    _click: function(e) {
      var text = getSelection();
      window.document.getSelection().removeAllRanges();
      this.props.marginalia.addAnnotation(text);
      this.hide();
    },
    render: function() {
      var state = this.state;
      var style = {
        display: state.visible ? "block" : "none",
        left: state.left,
        top: state.top
      };

      return (
          <div ref="popup"
               className="popup"
               style={style}>
            <div className="mask" onClick={this._click} onMouseDown={this._down}></div>
            <i className="fa fa-lg fa-pencil-square" />
          </div>
      );
    }
  });

  return Annotate;

});
