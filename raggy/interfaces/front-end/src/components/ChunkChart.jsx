import { ResponsiveBar } from '@nivo/bar';
import { RoundedBar } from './RoundedBar';
import './ChunkChart.css';

const CustomTooltip = ({ id, value, color, data, type }) => {
    const chunkTitleText = data.type === 'raptor' ? 'Raptor Summary' : 'Vanilla Chunk';
    let chunkId = data.type === 'raptor' ? data.raptorId : data.id;

    if (chunkId === undefined) {
        chunkId = 0;
    }
    return (
        <div
            style={{
                padding: '8px 10px',
                'word-wrap': 'break-word',
                background: 'white',
                border: `1px solid ${color}`,
                borderRadius: '2px',
                transform: `translate(70px, 0px)`,
                color: 'black',
                fontSize: '12px', // Smaller text size
                textAlign: 'left', // Align text to the left
                textAnchor: 'start', // Align text to the left
                maxWidth: '250px', // Max width of tooltip
            }}
        >
            <div>
                <strong>{`${chunkTitleText} ${chunkId}`}</strong>
                <span style={{ color, fontWeight: 'bold', marginLeft: '8px' }}>{`Score: ${value}`}</span>
            </div>
            <div>
                <span style={{ fontWeight: 'bold' }}>{`${data.text.substring(0, 300)}`}</span>
            </div>
        </div>

    );
}

export default function ChunkChart({
    chunks,
    selectedChunks,
    vanillaChunks,
    raptorChunks,
}) {
    // chunks have the following structure:
    // [
    //     { id: 0, score: 0.5, text: "This is the first chunk" },
    //     { id: 1, score: 0.6, text: "This is the second chunk" },
    //     ...

    // selectedChunks is an array of chunk ids that are selected
    // selectedChunks: [0, 1, 2, ...]

    // Ensure each data object includes the id
    const chunkData = chunks?.map(chunk => ({
        ...chunk,
        id: chunk.id.toString(), // Ensure id is a string for proper comparison
        text: chunk.text.toString(), // Ensure text is a string for proper comparison
    }));

    const amalgamatedChunks = vanillaChunks?.concat(raptorChunks);

    const amalgamatedChunkData = amalgamatedChunks?.map((chunk, index) => {
        const isRaptorChunk = raptorChunks.some(raptorChunk => raptorChunk.text === chunk.text);
        const chunkId = chunk.id === undefined ? '0' : chunk.id.toString();

        const id = isRaptorChunk ? (chunkId + vanillaChunks.length).toString() : chunkId.toString();

        if (id === undefined) {
            id = 0;
        }

        const type = isRaptorChunk ? 'raptor' : 'vanilla';
        return {
            ...chunk,
            id: id,
            text: chunk.text.toString(),
            type: type,
            raptorId: chunkId,
            score: (1 - Math.pow((1-chunk.score), 2)).toFixed(2)
        };
    }).sort((a, b) => b.score - a.score);




    // Ensure selectedChunks are all strings
    const selectedChunksStr = selectedChunks?.map((chunk) => {
        const isChunkSelected = selectedChunks.some(selectedChunk =>
            selectedChunk?.id?.toString() === chunk?.id?.toString() &&
            selectedChunk?.text?.toString() === chunk?.text?.toString()
        );
        if (isChunkSelected) {
            return chunk
        }
    });

    return (
        <div className="chart-box" style={{ background: "white", width: '100%', height: '300px' }}>
            {/* TODO: make the chart take the full width */}
            <ResponsiveBar
                data={amalgamatedChunkData}
                keys={['score']}
                indexBy="id"
                margin={{ top: 15, right: 40, bottom: 50, left: 60 }}
                padding={0.03}
                valueScale={{ type: 'linear' }}
                indexScale={{ type: 'band', round: true }}
                colors={({ data }) => { 
                    const isSelected = selectedChunksStr.some(selectedChunk => 
                    (selectedChunk?.id?.toString() === data?.id?.toString() ||
                    selectedChunk?.id?.toString() === data?.raptorId?.toString() ) &&
                    selectedChunk?.text?.toString() === data?.text?.toString() );
                    if(isSelected){
                        return 'rgba(69, 155, 255, 1)';
                    } else {
                        return 'rgba(69, 155, 255, 0.4)';
                    }
                }}
                axisTop={null}
                axisRight={null}
                axisBottom={{
                    tickSize: 1,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'chunk #',
                    legendPosition: 'middle',
                    legendOffset: 32,
                    truncateTickAt: 0
                }}
                axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'score',
                    legendPosition: 'middle',
                    legendOffset: -40,
                    truncateTickAt: 0
                }}
                labelSkipWidth={100} // Large enough to skip displaying labels inside the bars
                labelSkipHeight={100} // Large enough to skip displaying labels inside the bars
                labelTextColor={{
                    from: 'color',
                    modifiers: [
                        ['darker', 1.6]
                    ]
                }}
                role="application"
                ariaLabel="Nivo bar chart demo"
                barAriaLabel={e => e.id}
                tooltip={CustomTooltip} // Use custom tooltip component
                barComponent={RoundedBar} // Use custom bar component
                borderRadius={3} // Only round the first bar
            />
        </div>
    );
}
