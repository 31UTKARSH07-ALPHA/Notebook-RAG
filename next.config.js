/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ["unpdf", "@xenova/transformers"], 
  },

  webpack: (config) => {
    config.externals.push({
      'onnxruntime-node': 'commonjs onnxruntime-node'
    });

    config.module.rules.push({
      test: /onnxruntime-binding\.node$/,
      loader: 'null-loader'
    });

    return config;
  }
};

export default nextConfig;