/* -*- mode: js2; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js2-basic-offset: 2 -*- */
define(function (require) {
  'use strict';

  var _ = require("underscore");
  var $ = require("jquery");
  var React = require("react");
  var ReactDOM = require("react-dom");

  var TextLayerBuilder = require("../helpers/textLayerBuilder");

  var bounded = function(num) {
    return isFinite(num) ? num : null;
  };

  var VisibleArea = React.createClass({
    getInitialState: function() {
      return {
        mouseDown: false,
        offset: this.props.$viewer.scrollTop() / this.props.factor
      };
    },
    componentWillUnmount: function() {
      this.props.$viewer.off("scroll");
      var node = ReactDOM.findDOMNode(this);
      $(node.parentNode).off("mousedown mousemove");
      $("body").off("mouseup.minimap");
    },
    scrollTo: function(e, $minimap, $viewer) {
      var documentOffset = $minimap.offset().top;
      var offset = ((this.props.height / 2) + documentOffset);
      var y = e.pageY;
      this.setState({offset: y - offset});

      var scroll = (y - offset) * this.props.factor;
      $viewer.scrollTop(scroll);
    },
    componentDidMount: function() {
      var self = this;
      var $viewer =  this.props.$viewer;
      var node = ReactDOM.findDOMNode(this);
      var $minimap = $(node.parentNode);

      $viewer.on("scroll", function() {
        self.setState({offset: $viewer.scrollTop() / self.props.factor});
      });

      $("body").on("mouseup.minimap", function(e) {
        self.setState({mouseDown: false});
      });

      $minimap
        .on("mousemove", function(e) {
          if(self.state.mouseDown) {
            self.scrollTo(e, $minimap, $viewer);
          }
          return false;
        })
        .on("mousedown", function(e) {
          self.setState({mouseDown: true});
          // Jump to mousedown position
          self.scrollTo(e, $minimap, $viewer);
          return false;
        });
    },
    render: function() {
      var style = { height: bounded(this.props.height),
                    top: bounded(this.state.offset) };

      return (<div className="visible-area" style={style}></div>);
    }
  });

  var TextSegment = React.createClass({
    shouldComponentUpdate: function(nextProps, nextState) {
      return !_.isEqual(nextProps.annotation, this.props.annotation);
    },
    render: function() {
      var segment = this.props.segment;
      var annotation  = this.props.annotation;
        var style = {
          "top": (Math.ceil(segment.position) | 0) + "px",
          "height": (Math.ceil(segment.height) | 0) + "px"
        };

      if(annotation) {
        var color = annotation[0]["color"].join(",");
        style.backgroundColor = "rgb(" + color + ")";
      }
      return <div className="text-segment" style={style}></div>;
    }
  });

  var TextSegments = React.createClass({
    shouldComponentUpdate: function(nextProps, nextState) {
      return !_.isEqual(nextProps.annotations, this.props.annotations);
    },
    projectTextNodes: _.memoize(function(page, fingerprint, viewport, factor) {
      // The basic idea here is using a sweepline to
      // project the 2D structure of the PDF onto a 1D minimap
      var self = this;
      var content = page.get("content");

      var textLayerBuilder = new TextLayerBuilder({viewport: viewport});

      var nodes = content.items.map(function(geom, idx) {
        var style = textLayerBuilder.calculateStyles(geom, content.styles[geom.fontName]);

        return {
          height: parseInt(style.fontSize, 10) / factor,
          position: parseInt(style.top, 10) / factor,
          idx: [idx + ""]
        };
      });

      var segments = [];
      var sortedByPosition = _.sortBy(nodes, function(n) { return n.position; });
      for(var i = 0; i < sortedByPosition.length; i++) {
        var node = sortedByPosition[i];
        var prevSegment = segments.slice(-1).pop(); // peek
        if(segments.length === 0) {
          segments.push(node);
          continue;
        }

        if((prevSegment.position + prevSegment.height) >= node.position) {
          prevSegment = segments.pop();
          var nextHeight =  prevSegment.height +
                ((node.height + node.position) - (prevSegment.height + prevSegment.position));
          var nextIdx = _.union(prevSegment.idx, [node.idx + ""]);

          var nextSegment = {
            height: nextHeight,
            position: prevSegment.position,
            idx: nextIdx
          };
          segments.push(_.extend(node, nextSegment));
        } else {
          segments.push(node);
        }
      }
      return segments;
    }, function(page, fingerprint) { return fingerprint + page.get("raw").pageIndex; }) ,
    render: function() {
      var page = this.props.page;
      var raw = page.get("raw");

      var factor = this.props.factor;
      var annotations = this.props.annotations;

      var viewport = raw.getViewport(1.0);
      var pageWidthScale = this.props.$viewer.width() / viewport.width;
      viewport = raw.getViewport(pageWidthScale);

      var fingerprint = raw.transport.pdfDocument.pdfInfo.fingerprint; // hack

      var textNodes = this.projectTextNodes(page, fingerprint, viewport, factor);

      var annotated = _.keys(annotations);
      var textSegments = textNodes.map(function(segment, idx) {
        var ann = _.first(_.intersection(annotated, segment.idx));
        return <TextSegment key={idx} segment={segment} annotation={annotations[ann]} />;
      });

      return <div>{textSegments}</div>;
    }
  });

  var PageSegment = React.createClass({

    render: function() {
      var page = this.props.page;
      var textSegments = null;
      if(page.get("state") >= RenderingStates.HAS_CONTENT) {
        textSegments = (
            <TextSegments page={page}
                          annotations={this.props.annotations}
                          factor={this.props.factor}
                          $viewer={this.props.$viewer} />);
      }
      return <div className="minimap-node" style={this.props.style}>{textSegments}</div>;
    }
  });


  // FIXME really really hacky way of maintaining state
  var lastHeight = {};

  var Minimap = React.createClass({
    render: function() {
      var $viewer = this.props.$viewer;
      if(!$viewer) return null; // wait for viewer to mount

      var pdf = this.props.pdf;
      var fingerprint = pdf.get("fingerprint");
      var pages = pdf.get("pages");
      var numPages = pages.length;

      // We assume that each page has the same height.
      // This is not true sometimes, but often enough for academic papers.
      var $firstPage = $viewer.find(".page").first();
      var pageHeight = $firstPage.height();
      // We store this because React stateful rendering might be out of sync...
      if(lastHeight[fingerprint] >= pageHeight) {
        pageHeight = lastHeight[fingerprint];
      } else {
        lastHeight[fingerprint] = pageHeight;
      }
      var totalHeight = pageHeight * numPages;

      var offset = $viewer.offset().top;
      var viewerHeight = $viewer.height();
      var factor = totalHeight / (viewerHeight - offset);

      var height = viewerHeight / factor;
      var annotations = this.props.annotations;

      var pageElements = pages.map(function(page, pageIndex) {
        var style = {height: bounded((totalHeight / numPages) / factor)};
        return <PageSegment key={pageIndex}
                            page={page}
                            $viewer={$viewer}
                            factor={factor}
                            annotations={annotations[pageIndex]}
                            style={style} />;
      });

      return (<div className="minimap">
                <VisibleArea height={height} $viewer={$viewer} factor={factor} />
                {pageElements}
              </div>);
    }
  });

  return Minimap;
});
