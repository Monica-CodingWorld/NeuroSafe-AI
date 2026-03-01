from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
alerts = []

HTML = """
<h2>NeuroSafe Dashboard</h2>
<ul>
{% for a in alerts %}
<li>{{a}}</li>
{% endfor %}
</ul>
<script>
setTimeout(()=>location.reload(),2000);
</script>
"""

@app.route("/alert", methods=["POST"])
def alert():
    data = request.json
    alerts.append(data)
    return jsonify({"status":"ok"})

@app.route("/")
def home():
    return render_template_string(HTML, alerts=alerts)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)