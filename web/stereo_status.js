import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "DukeStereo.Status",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Duke Stereo Prepared Sequence") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            this.dukeStatus = this.addWidget("text", "Status", "Warte auf fertige Tiefe", () => {}, {
                serialize: false,
            });
            this.dukeStatus.disabled = true;
            this.dukeStatus.computeSize = () => [this.size[0], 48];
            this.dukeStatus.draw = function (ctx, node, width, y) {
                ctx.save();
                ctx.fillStyle = "#ddd";
                ctx.font = "13px sans-serif";
                const lines = String(this.value).split("\n");
                lines.forEach((line, i) => ctx.fillText(line, 12, y + 18 + i * 20, width - 24));
                ctx.restore();
            };
            this.setSize(this.computeSize());
        };
    },
});

api.addEventListener("duke_stereo_status", ({ detail }) => {
    const node = app.graph?.getNodeById(detail.node);
    if (!node?.dukeStatus) return;
    node.dukeStatus.value = detail.text;
    app.graph.setDirtyCanvas(true, false);
});
