var colors = d3.scaleOrdinal(d3.schemeCategory10);

var svg = d3.select("svg");
var width = svg.attr("width");
var height = svg.attr("height");
var node;
var link;

// syle for arrow head in the svg...
svg
  .append("defs")
  .append("marker")
  .attrs({
    id: "arrowhead",
    viewBox: "-0 -5 10 10",
    refX: 13,
    refY: 0,
    orient: "auto",
    markerWidth: 13,
    markerHeight: 13,
    xoverflow: "visible",
  })
  .append("svg:path")
  .attr("d", "M 0,-5 L 10 ,0 L 0,5")
  .attr("fill", "#999")
  .style("stroke", "none");

var svg_simulation = d3
  .forceSimulation()
  .force("charge", d3.forceManyBody().strength(-30))
  .force("r", d3.forceRadial(200, width / 2, height / 2))
  .force(
    "link",
    d3
      .forceLink()
      .distance(80)
      .strength(0.01)
      .iterations(10)
      .id(function (d) {
        return d.id;
      })
  );
//.force("x", d3.forceX(width/2).strength(0.04))
//.force("y", d3.forceY(height/2).strength(0.06));

var svg_graph = JSON.parse(JSON.stringify(window._data_graph));
(function (error, graph) {
  if (error) throw error;
  graph.nodes.forEach(function (d) {
    d.x = width * Math.random();
    d.y = height * Math.random();
  });
  update(graph.links, graph.nodes);
})(null, svg_graph);

var edgelabels;

function update(links, nodes) {
  link = svg
    .selectAll(".link")
    .data(links)
    .enter()
    .append("line")
    /* transform that into css:
     line.link {
        marker-end: url(#arrowhead);
     }
     */
    .attrs({
      class: "link",
      stroke: "gray",
      "marker-end": "url(#arrowhead)",
    });

  //link.append("title")
  //    .text(function (d) {return d.type;});

  edgelabels = svg
    .selectAll(".edgelabel")
    .data(links)
    .enter()
    .append("text")
    .style("pointer-events", "none")
    .attrs({
      class: "edgelabel",
      "font-size": 10,
      fill: "#aaa",
    });

  edgelabels
    .append("textPath")
    .attr("xlink:href", function (d, i) {
      return "#edgepath" + i;
    })
    .style("text-anchor", "middle")
    .style("pointer-events", "none")
    .attr("startOffset", "50%")
    .text(function (d) {
      return d.type;
    });

  node = svg
    .selectAll(".node")
    .data(nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .call(
      d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

  node
    .append("a")
    .attr("xlink:href", function (node) {
      return node.url;
    })
    .append("circle")
    .attr("r", function (node) {
      return node.val;
    })
    .style("fill", function (d, i) {
      return colors(d.mod);
    })
    .style("stroke", "#FFF");

  node.append("title").text(function (d) {
    return d.label;
  });

  node
    .append("text")
    //.attr("dx", function(node){return node.val+1})
    .attr("dy", +10)
    .attr("opacity", "0.5")
    .text(function (d) {
      arr = d.label.split(".");
      return arr[arr.length - 1];
    });

  svg_simulation.nodes(nodes).on("tick", ticked);

  svg_simulation.force("link").links(links);
}

function ticked() {
  link
    .attr("x1", function (d) {
      return d.source.x;
    })
    .attr("y1", function (d) {
      return d.source.y;
    })
    .attr("x2", function (d) {
      return d.target.x;
    })
    .attr("y2", function (d) {
      return d.target.y;
    });

  node.attr("transform", function (d) {
    return "translate(" + d.x + ", " + d.y + ")";
  });

  edgelabels.attr("transform", function (d) {
    if (d.target.x < d.source.x) {
      var bbox = this.getBBox();

      rx = bbox.x + bbox.width / 2;
      ry = bbox.y + bbox.height / 2;
      return "rotate(180 " + rx + " " + ry + ")";
    } else {
      return "rotate(0)";
    }
  });
}

function dragstarted(d) {
  if (!d3.event.active) svg_simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) svg_simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
