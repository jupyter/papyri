var canvas = document.querySelector("canvas"),
  context = canvas.getContext("2d"),
  width = canvas.width,
  height = canvas.height;
var searchRadius = 40;

var color = d3.scaleOrdinal().range(d3.schemeCategory10);

var canvas_simulation = d3
  .forceSimulation()
  .force("charge", d3.forceManyBody().strength(-100))
  .force(
    "link",
    d3
      .forceLink()
      .distance(30)
      .strength(0.1)
      .iterations(1)
      .id(function (d) {
        return d.id;
      })
  )
  //.force("x", d3.forceX().strength(0.1))
  //.force("center", d3.forceCenter(width / 2, height / 2))
  .force("r", d3.forceRadial(150));
//.force("y", d3.forceY().strength(0.2));

var canvas_graph = JSON.parse(JSON.stringify(window._data_graph));

(function () {
  var pages = d3
    .nest()
    .key(function (d) {
      return d.mod;
    })
    .entries(canvas_graph.nodes)
    .sort(function (a, b) {
      return b.values.length - a.values.length;
    });

  var c = color.domain(
    pages.map(function (d) {
      return d.key;
    })
  );

  canvas_simulation.nodes(canvas_graph.nodes).on("tick", ticked);

  canvas_simulation.force("link").links(canvas_graph.links);

  d3.select(canvas)
    .on("mousemove", mousemoved)
    .call(
      d3
        .drag()
        .container(canvas)
        .subject(dragsubject)
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

  function ticked() {
    context.clearRect(0, 0, width, height);
    context.save();
    context.translate(width / 2, height / 2);

    context.beginPath();
    canvas_graph.links.forEach(drawLink);
    context.strokeStyle = "#aaa";
    context.stroke();
    var margin = 10;

    pages.forEach(function (page) {
      context.beginPath();
      page.values.forEach(drawNode);
      context.fillStyle = color(page.key);
      context.fill();
    });

    context.restore();
  }

  function dragsubject() {
    return canvas_simulation.find(
      d3.event.x - width / 2,
      d3.event.y - height / 2,
      searchRadius
    );
  }

  function mousemoved() {
    var a = this.parentNode,
      m = d3.mouse(this),
      d = canvas_simulation.find(
        m[0] - width / 2,
        m[1] - height / 2,
        searchRadius
      );
    if (!d) return a.removeAttribute("href"), a.removeAttribute("title");
    //a.setAttribute("href", "http://bl.ocks.org/" + (d.user ? d.user + "/" : "") + d.id);
    a.setAttribute("title", d.label);
  }
})();

function dragstarted() {
  if (!d3.event.active) canvas_simulation.alphaTarget(0.3).restart();
  d3.event.subject.fx = d3.event.subject.x;
  d3.event.subject.fy = d3.event.subject.y;
}

function dragged() {
  d3.event.subject.fx = d3.event.x;
  d3.event.subject.fy = d3.event.y;
}

function dragended() {
  if (!d3.event.active) canvas_simulation.alphaTarget(0);
  d3.event.subject.fx = null;
  d3.event.subject.fy = null;
}

function drawLink(d) {
  context.moveTo(d.source.x, d.source.y);
  context.lineTo(d.target.x, d.target.y);
  //context.moveTo(d.x + 3, d.y);
  //context.arc((d.target.x*9+d.source.x)/10, (d.target.y*9+d.source.y)/10, 2, 0, 2 * Math.PI);
}

function drawNode(d) {
  context.moveTo(d.x + 5, d.y);
  context.arc(d.x, d.y, d.val, 0, 2 * Math.PI);
}
