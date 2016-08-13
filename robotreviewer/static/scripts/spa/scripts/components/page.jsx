/* -*- mode: js2; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js2-basic-offset: 2 -*- */
define(function (require) {
  'use strict';

  var _ = require("underscore");
  var $ = require("jquery");

  var React = require("react");
  var ReactDOM = require("react-dom");
  var TextLayerBuilder = require("../helpers/textLayerBuilder");

  var getOutputScale = function(ctx) {
    var devicePixelRatio = window.devicePixelRatio || 1;
    var backingStoreRatio = ctx.webkitBackingStorePixelRatio ||
        ctx.mozBackingStorePixelRatio ||
        ctx.msBackingStorePixelRatio ||
        ctx.oBackingStorePixelRatio ||
        ctx.backingStorePixelRatio || 1;
    var pixelRatio = devicePixelRatio / backingStoreRatio;
    return {
      sx: pixelRatio,
      sy: pixelRatio,
      scaled: pixelRatio !== 1
    };
  }

  var TextNode = React.createClass({
    triggerHighlight: function(uuid) {
      $(window).trigger("highlight", uuid);
    },
    shouldComponentUpdate: function(nextProps) {
      return !_.isEqual(nextProps.annotations, this.props.annotations);
    },
    render: function() {
      var p = this.props;
      var self = this;
      var annotations;
      if(this.props.annotations) {
        annotations = this.props.annotations;
      }
      var o = p.textLayerBuilder.createAnnotatedElement(p.item, p.styles, annotations);

      if(o.isWhitespace) { return null; }

      var content;
      if(o.spans) {
        var uuids = _.pluck(annotations, "uuid");
        content = o.spans.map(function(s,i) {
          if(!s) return null;

          var highlight = function(e) {
            self.triggerHighlight(s.uuid);
          };

          return <span key={i}>
                   <span className="pre">{s.pre}</span>
                   <span className="annotated"
                         style={s.style}
                         data-color={s.color}
                         onMouseEnter={highlight}
                         onMouseLeave={highlight}
                         data-uuid={uuids}>{s.content}</span>
                   <span className="post">{s.post}</span>
                  </span>;
        });
      } else {
        content = o.textContent;
      }

      return <div style={o.style} dir={o.dir}>{content}</div>;
    }
  });

  var TextLayer = React.createClass({
    shouldComponentUpdate: function(nextProps, nextState) {
      return !_.isEqual(nextProps.annotations, this.props.annotations);
    },
    getTextLayerBuilder: function(viewport) {
      return new TextLayerBuilder({viewport: viewport});
    },
    render: function() {
      var page  = this.props.page;
      var self = this;
      var content = page.get("content");
      var textLayerBuilder = this.getTextLayerBuilder(this.props.viewport);
      var annotations = this.props.annotations;
      var styles = content.styles;
      var textNodes = content.items.map(function (item,i) {
        return <TextNode key={i}
                         item={item}
                         annotations={annotations[i]}
                         styles={styles}
                         textLayerBuilder={textLayerBuilder} />;
      });
      return (
        <div style={this.props.dimensions} className="textLayer">
          {textNodes}
        </div>);
    }
  });

  var Page = React.createClass({
    getInitialState: function() {
      return {
        isRendered: false,
        renderingState: RenderingStates.INITIAL
      };
    },
    componentWillReceiveProps: function(nextProps) {
      this.setState({renderingState: nextProps.page.get("state")});
    },
    drawPage: function(page) {
      var self = this;
      var container = ReactDOM.findDOMNode(this);
      var canvas = this.refs.canvas;
      var ctx = canvas.getContext("2d");

      var viewport = page.getViewport(1.0);
      var pageWidthScale = container.clientWidth / viewport.width;
      viewport = page.getViewport(pageWidthScale);

      var outputScale = getOutputScale(ctx);

      canvas.width = (Math.floor(viewport.width) * outputScale.sx) | 0;
      canvas.height = (Math.floor(viewport.height) * outputScale.sy) | 0;
      canvas.style.width = Math.floor(viewport.width) + 'px';
      canvas.style.height = Math.floor(viewport.height) + 'px';

      this.setState({
        viewport: viewport,
        dimensions: { width: canvas.width + "px",
                      height: canvas.height + "px" }});

      ctx._scaleX = outputScale.sx;
      ctx._scaleY = outputScale.sy;

      if (outputScale.scaled) {
        ctx.scale(outputScale.sx, outputScale.sy);
      }
      var renderContext = {
        canvasContext: ctx,
        viewport: viewport
      };

      // Store a refer to the renderer
      var pageRendering = page.render(renderContext);
      // Hook into the pdf render complete event
      var completeCallback = pageRendering._internalRenderTask.callback;
      pageRendering._internalRenderTask.callback = function (error) {
        completeCallback.call(this, error);
        self.setState({isRendered: true});
      };
    },
    componentDidUpdate: function(prevProps, prevState) {
      if(this.state.renderingState >= RenderingStates.HAS_PAGE && !this.state.viewport) {
        this.drawPage(this.props.page.get("raw"));
      }
    },
    render: function() {
      var textLayer = null;
      var renderingState = this.state.renderingState;
      var isLoading = !this.state.isRendered;

      if(this.state.viewport && renderingState >= RenderingStates.HAS_CONTENT) {
        textLayer = <TextLayer dimensions={this.state.dimensions}
                               viewport={this.state.viewport}
                               annotations={this.props.annotations}
                               page={this.props.page} />;
      }

      var loader = require.toUrl(".") + "/../../img/loading-spin.svg";
      return (
        <div className="page">
          <div className="loading" style={{opacity: isLoading ? 1 : 0}}>
            <img src={loader} viewBox="0 0 24 24" width="24" height="24" />
          </div>
          <canvas ref="canvas" />
          {textLayer}
        </div>);
    }
  });

  return Page;

});
